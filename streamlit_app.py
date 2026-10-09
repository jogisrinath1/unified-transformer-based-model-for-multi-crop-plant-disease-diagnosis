import os
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import torch
from torch import nn
from PIL import Image
import timm
import torchvision.transforms as T
import textwrap
from io import BytesIO
import base64
import streamlit.components.v1 as components
import re

try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False

try:
    import google.generativeai as genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False

# --------------------
# Config and constants
# --------------------
DATA_ROOT = Path('PlantVillageDataset/train_val_test/test')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.JPG', '.JPEG', '.PNG'}
IM_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

GEMINI_API_KEY = 'AIzaSyDMHHkh6JxPkAdn3TlZrm8gPBPfuK4DV-s'
GEMINI_MODEL_PRIMARY = 'gemini-2.5-flash'
GEMINI_MODEL_FALLBACKS = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-flash-8b']


def build_treatment_prompt(
    disease_label: str,
    top_preds: List[Tuple[str, float]],
    location: Optional[str] = None,
    response_language: Optional[str] = None,
) -> str:
    top_lines = '\n'.join([f"- {lbl}: {p*100:.2f}%" for lbl, p in top_preds[:5]])
    loc = f" in {location}" if location else ""
    lang_directive = f"Respond in {response_language}." if response_language else "Respond in English."
    return textwrap.dedent(f"""
    You are an agronomy assistant. The model predicted a plant disease{loc}.
    Task: Recommend practical treatment options and safe chemicals.

    Predicted disease: {disease_label}
    Top probabilities:\n{top_lines}

    Provide:
    - Diagnosis confidence considerations and how to double-check.
    - Chemical treatment: 2-4 recommended active ingredients (not brands), typical field rates, pre-harvest interval, and FRAC/IRAC mode-of-action to support rotation.
    - Non-chemical control: cultural practices, sanitation, resistant varieties, irrigation and pruning tips.
    - Safety: PPE, re-entry interval, pollinator care.
    - A short, stepwise 7-day action plan.

    Constraints:
    - Keep to concise bullet points.
    - Prefer globally recognized actives. Note regional restrictions if relevant.
    - Include a brief disclaimer to check local labels and regulations.
    - {lang_directive}
    - If using a non-English language, add English technical terms in parentheses when helpful.
    """)


@st.cache_resource(show_spinner=False)
def get_gemini_model(api_key: str, model_name: str):
    if not _HAS_GENAI:
        raise RuntimeError('google-generativeai package not available. Add it to requirements and reinstall.')
    if not api_key:
        raise RuntimeError('Missing Google API key.')
    genai.configure(api_key=api_key)
    tried = [model_name] + [m for m in GEMINI_MODEL_FALLBACKS if m != model_name]
    last_err = None
    for m in tried:
        try:
            return genai.GenerativeModel(m)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'Failed to initialize Gemini model. Tried: {", ".join(tried)}. Last error: {last_err}')


def list_pt_candidates() -> List[Path]:
    candidates = []
    for p in [Path('vit_best.pt'), Path('artifacts/vit_best.pt'), Path('artifacts/hybrid_effvit_best.pt'),
              Path('artifacts/first_model_resnet.pt'), Path('artifacts/seccont_model_resnet.pt')]:
        if p.exists():
            candidates.append(p)
    for p in Path('.').glob('*.pt'):
        if p not in candidates:
            candidates.append(p)
    for p in Path('artifacts').glob('*.pt') if Path('artifacts').exists() else []:
        if p not in candidates:
            candidates.append(p)
    return candidates


# --------------------
# Hybrid model wrapper
# --------------------
class HybridEffNetViT(nn.Module):
    def __init__(self, num_classes: int, eff_name: str = 'efficientnet_b0', vit_name: str = 'vit_small_patch16_224'):
        super().__init__()
        self.eff = timm.create_model(eff_name, pretrained=False, num_classes=0, global_pool='avg')
        self.vit = timm.create_model(vit_name, pretrained=False, num_classes=0, global_pool='avg')
        in_feats = self.eff.num_features + self.vit.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Dropout(0.2),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        f1 = self.eff(x)
        f2 = self.vit(x)
        feat = torch.cat([f1, f2], dim=1)
        return self.head(feat)


# --------------------
# Model loading
# --------------------
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model') or ckpt.get('state_dict')
    classes = ckpt.get('classes', None)
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes' list.")
    num_classes = len(classes)

    keys = list(state_dict.keys())
    is_hybrid = any(k.startswith('eff.') for k in keys) and any(k.startswith('vit.') for k in keys)

    last_err: Optional[Exception] = None
    model = None
    arch_name = ''

    if not is_hybrid:
        for name in ['vit_base_patch16_224', 'vit_small_patch16_224']:
            try:
                m = timm.create_model(name, pretrained=False, num_classes=num_classes)
                m.load_state_dict(state_dict, strict=True)
                model = m
                arch_name = name
                break
            except Exception as e:
                last_err = e
    else:
        for vit_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
            try:
                m = HybridEffNetViT(num_classes=num_classes, eff_name='efficientnet_b0', vit_name=vit_name)
                m.load_state_dict(state_dict, strict=True)
                model = m
                arch_name = f'hybrid: eff_b0 + {vit_name}'
                break
            except Exception as e:
                last_err = e

    if model is None:
        try:
            m = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
            m.load_state_dict(state_dict, strict=False)
            model = m
            arch_name = 'vit_base_patch16_224 (non-strict)'
        except Exception:
            raise last_err or RuntimeError('Unable to construct model from checkpoint')

    model.to(device)
    model.eval()
    return model, classes, arch_name


# --------------------
# Preprocessing & Prediction
# --------------------
@st.cache_resource(show_spinner=False)
def get_transform():
    return T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])


@torch.inference_mode()
def predict(model: nn.Module, img: Image.Image, classes: List[str], device: torch.device, topk: int = 5):
    tfm = get_transform()
    x = tfm(img.convert('RGB')).unsqueeze(0)
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    topk = min(topk, probs.numel())
    conf, idx = torch.topk(probs, k=topk)
    return [(classes[i], float(conf[j])) for j, i in enumerate(idx.tolist())]


# --------------------
# Test set utilities
# --------------------
@st.cache_resource(show_spinner=False)
def index_test_images(root: Path):
    idx = {}
    if root.exists():
        for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
            files = [p for p in cls_dir.rglob('*') if p.suffix in IMG_EXTS]
            if files:
                idx[cls_dir.name] = sorted(files)
    return idx


# --------------------
# Enhanced TTS Functions
# --------------------
def clean_text_for_tts(md_text: str) -> str:
    if not md_text:
        return md_text
    text = md_text
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)
    text = text.replace('**', '').replace('*', '')
    text = text.replace('__', '').replace('_', ' ')
    lines = []
    for line in text.splitlines():
        line = re.sub(r'^\s*#{1,6}\s*', '', line)
        line = re.sub(r'^\s*>\s*', '', line)
        line = re.sub(r'^\s*[-*+]\s+', '', line)
        if re.match(r'^\s*[-|:\s]*$', line):
            continue
        line = line.replace('|', ', ')
        if line.strip():
            lines.append(line.strip())
    text = '. '.join(lines)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = text.replace(':', '. ')
    text = text.replace(';', '. ')
    return text.strip()


def synthesize_tts(text: str, lang_code: str) -> Optional[bytes]:
    if not _HAS_GTTS or not text.strip():
        return None
    try:
        if len(text) > 5000:
            text = text[:5000] + "..."
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None


def create_audio_player(audio_bytes: bytes, auto_play: bool = True) -> None:
    if not audio_bytes:
        return
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_html = f"""
    <div style="margin: 10px 0;">
        <audio {'autoplay' if auto_play else ''} controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
    components.html(audio_html, height=60)


def display_treatment_section(pred_label: str, preds: List[Tuple[str, float]], user_location: Optional[str]):
    st.markdown('---')
    st.subheader('🎯 Treatment Suggestions (Gemini AI)')
    if not _HAS_GENAI:
        st.info('📦 Install google-generativeai to enable AI treatment suggestions.')
        return
    col1, col2 = st.columns([2, 1])
    with col1:
        language_options = [
            ('English', 'en'), ('Hindi', 'hi'), ('Telugu', 'te'), ('Tamil', 'ta'),
            ('Kannada', 'kn'), ('Malayalam', 'ml'), ('Marathi', 'mr'), ('Gujarati', 'gu'),
            ('Bengali', 'bn'), ('Punjabi', 'pa'), ('Urdu', 'ur'), ('Spanish', 'es'),
            ('French', 'fr'), ('German', 'de'), ('Arabic', 'ar'), ('Indonesian', 'id'),
            ('Swahili', 'sw'), ('Portuguese', 'pt'), ('Russian', 'ru'), ('Chinese', 'zh-CN'),
            ('Japanese', 'ja'), ('Korean', 'ko')
        ]
        selected_lang = st.selectbox(
            '🌍 Response & Audio Language',
            options=language_options,
            index=0,
            format_func=lambda x: x[0],
            help="Select language for both text response and audio narration"
        )
        lang_name, lang_code = selected_lang
    with col2:
        enable_audio = st.checkbox('🔊 Enable Audio', value=True, help="Auto-generate audio narration")
        auto_play = st.checkbox('▶️ Auto-play', value=True, help="Automatically play audio when generated")
    if st.button('🚀 Get Treatment Suggestions', type="primary"):
        with st.spinner(f'🤖 Generating treatment recommendations in {lang_name}...'):
            try:
                gemini_model = get_gemini_model(GEMINI_API_KEY, GEMINI_MODEL_PRIMARY)
                prompt = build_treatment_prompt(pred_label, preds, user_location, response_language=lang_name)
                response = gemini_model.generate_content(prompt)
                response_text = (
                    getattr(response, 'text', None) or
                    (response.candidates[0].content.parts[0].text if hasattr(response, 'candidates') and response.candidates else '')
                )
                if not response_text:
                    st.warning('⚠️ Received empty response from Gemini AI.')
                    return
                st.markdown("### 📋 Treatment Recommendations")
                st.markdown(response_text)
                if enable_audio and _HAS_GTTS:
                    with st.spinner('🎵 Generating audio narration...'):
                        clean_text = clean_text_for_tts(response_text)
                        if clean_text:
                            audio_bytes = synthesize_tts(clean_text, lang_code)
                            if audio_bytes:
                                st.success(f'🎵 Audio generated successfully! ({len(audio_bytes)} bytes)')
                                create_audio_player(audio_bytes, auto_play=auto_play)
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=audio_bytes,
                                    file_name=f"treatment_audio_{lang_code}.mp3",
                                    mime="audio/mpeg"
                                )
                            else:
                                st.warning('⚠️ Unable to generate audio for the selected language.')
                        else:
                            st.warning('⚠️ No suitable text found for audio generation.')
                elif enable_audio and not _HAS_GTTS:
                    st.info('📦 Install gTTS to enable audio narration.')
            except Exception as e:
                st.error(f'❌ Error generating recommendations: {str(e)}')


# --------------------
# Main Streamlit UI
# --------------------
st.set_page_config(
    page_title='PlantVillage: AI Plant Disease Diagnosis',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.title('🌱 Unified Transformer-Based Plant Disease Diagnosis')
st.markdown('*Load your trained checkpoint (.pt), predict on test images or uploads, and get AI-powered treatment recommendations with audio narration.*')

st.sidebar.header('🔧 Model Configuration')
candidates = list_pt_candidates()
default_ckpt = str(candidates[0]) if candidates else 'vit_best.pt'
ckpt_path = st.sidebar.text_input('📁 Checkpoint path (.pt)', value=default_ckpt)
use_cuda = st.sidebar.checkbox('🚀 Use CUDA if available', value=torch.cuda.is_available())

st.sidebar.markdown('---')
st.sidebar.header('🌍 Location Settings')
user_location = st.sidebar.text_input(
    '📍 Your Location (optional)',
    value='',
    help='Country/region to tailor treatment recommendations',
    placeholder='e.g., India, California, etc.'
)

st.sidebar.markdown('---')
st.sidebar.header('🤖 AI Models')
st.sidebar.info(f'**Gemini Model:** {GEMINI_MODEL_PRIMARY}')
st.sidebar.caption('Fallbacks: ' + ', '.join(GEMINI_MODEL_FALLBACKS))

device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

model = None
classes = None
arch_name = ''

if ckpt_path and Path(ckpt_path).exists():
    with st.spinner('🔄 Loading model...'):
        try:
            model, classes, arch_name = load_model(ckpt_path, device)
            st.sidebar.success(f'✅ Model loaded: {arch_name}')
            st.sidebar.info(f'📊 Classes: {len(classes)}')
        except Exception as e:
            st.sidebar.error(f'❌ Failed to load model: {e}')
else:
    st.sidebar.warning('⚠️ Checkpoint not found. Please provide a valid path.')

st.header('📤 Input Selection')
mode = st.radio('Choose input mode:', ['🗂️ Test set browser', '📤 Upload image'])

if mode == '🗂️ Test set browser':
    idx = index_test_images(DATA_ROOT)
    if not idx:
        st.warning(f'⚠️ No test images found under {DATA_ROOT}')
    else:
        col1, col2 = st.columns(2)
        with col1:
            cls_choice = st.selectbox('📂 Select Class', list(idx.keys()))
        img_choice = None
        if cls_choice:
            files = idx[cls_choice]
            img_strs = [str(p.relative_to(DATA_ROOT)) for p in files]
            with col2:
                img_sel = st.selectbox('🖼️ Select Image', img_strs)
            if img_sel:
                img_choice = DATA_ROOT / img_sel
        if img_choice and model is not None:
            img = Image.open(img_choice)
            st.image(img, caption=f'📷 Test image: {img_choice.name}', use_column_width=True)
            preds = predict(model, img, classes, device, topk=5)
            pred_label, pred_conf = preds[0]
            correct = (pred_label == cls_choice)
            st.header('🎯 Prediction Results')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('🏷️ Prediction', pred_label, delta=f'{pred_conf*100:.2f}%')
            with col2:
                st.metric('✅ Accuracy', 'Correct' if correct else 'Incorrect', delta='✓' if correct else '✗')
            with col3:
                st.metric('🎯 Confidence', f'{pred_conf*100:.1f}%', delta=None)
            with st.expander('📊 View Top-5 Predictions', expanded=False):
                for i, (lbl, p) in enumerate(preds, 1):
                    st.write(f'{i}. **{lbl}**: {p*100:.2f}%')
            display_treatment_section(pred_label, preds, user_location)
else:
    uploaded = st.file_uploader(
        '📤 Upload an image',
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF"
    )
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption=f'📷 Uploaded: {uploaded.name}', use_column_width=True)
        if model is not None:
            preds = predict(model, img, classes, device, topk=5)
            pred_label, pred_conf = preds[0]
            st.header('🎯 Prediction Results')
            col1, col2 = st.columns(2)
            with col1:
                st.metric('🏷️ Prediction', pred_label, delta=f'{pred_conf*100:.2f}%')
            with col2:
                st.metric('🎯 Confidence', f'{pred_conf*100:.1f}%', delta=None)
            with st.expander('📊 View Top-5 Predictions', expanded=False):
                for i, (lbl, p) in enumerate(preds, 1):
                    st.write(f'{i}. **{lbl}**: {p*100:.2f}%')
            display_treatment_section(pred_label, preds, user_location)
        else:
            st.warning('⚠️ Please load a model checkpoint to run inference.')

st.markdown('---')
st.caption('💡 **Tip:** Ensure your checkpoint includes the class list ("classes") for proper model loading.')
st.caption('🔧 **Note:** Audio feature requires gTTS package. Treatment suggestions require google-generativeai package.')
