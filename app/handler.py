"""
에고북 유해표현 검출 API - AWS Lambda 핸들러
KcElectra 모델 (F1-Score: 0.9928) 사용
"""

import os
import json
import re
import torch
import numpy as np

# 환경변수 설정
os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache/huggingface"
os.environ["HF_HOME"] = "/tmp/cache/huggingface"
os.environ["TORCH_HOME"] = "/tmp/cache/torch"

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap 라이브러리 없음")

# ============================================================
# 전역 변수
# ============================================================
MODEL_PATH = "/var/task/models/KcElectra_best_0.9928"
model = None
tokenizer = None
pipe = None
device = None

ID2LABEL = {0: "비유해", 1: "유해표현포함"}

SLANG_DICT = {
    'ㅅㅂ', 'ㅆㅂ', 'ㅂㅅ', 'ㅄ', 'ㅈㄹ', 'ㅁㅊ', 'ㅅㄲ', 'ㅆㄲ',
    'ㄱㅅㄲ', 'ㄴㅁ', 'ㄲㅈ', 'ㅈㄴ', 'ㄷㅊ', 'ㅈㅂ', 'ㅗ', 'ㅗㅗ',
    'ㄱㅅㄹ', 'ㅂㄹ', 'ㅍㅅ', 'ㅆㅍ', 'ㅈㅈ',
    'ㅅㅍ', 'ㅂㅁ', 'ㅊㄴ', 'ㅍㅈ', 'ㄴㄱㅁ', 'ㅆㄹㄱ',
}

def check_slang(text: str) -> list:
    """초성 욕설 검사"""
    found = []
    for slang in SLANG_DICT:
        if slang in text:
            found.append(slang)
    return found

def load_model():
    """모델 로드 (최초 1회만)"""
    global model, tokenizer, pipe, device
    
    if model is not None:
        return
    
    print("[INFO] 모델 로딩 시작...")
    
    # 캐시 디렉토리 생성
    os.makedirs("/tmp/cache/huggingface", exist_ok=True)
    os.makedirs("/tmp/cache/torch", exist_ok=True)
    
    device = "cpu"  
    print(f"[INFO] 디바이스: {device}")
    
    # 모델 & 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True 
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model.eval()
    
    # SHAP용 파이프라인
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1 
    )


def extract_harmful_tokens(text: str) -> dict:
    """SHAP 유해 판정에 영향 준 토큰 추출"""
    if not SHAP_AVAILABLE or pipe is None:
        return {"harmful_tokens": {}}
    
    try:
        explainer = shap.Explainer(pipe)
        shap_values = explainer([text])
        shap_for_text = shap_values[0, :, 1]
        
        values = np.round(shap_for_text.values, 4)
        tokens = shap_for_text.data
        cleaned_tokens = [re.sub(r'^[Ġ▁]', '', str(t)).strip() for t in tokens]
        
        EXCLUDE_TOKENS = {
            '[CLS]', '[SEP]', '[PAD]', '[UNK]', '',
            'ㅋ', 'ㅋㅋ', 'ㅋㅋㅋ', 'ㅋㅋㅋㅋ',
            'ㅎ', 'ㅎㅎ', 'ㅎㅎㅎ', 'ㅎㅎㅎㅎ',
            'ㅠ', 'ㅠㅠ', 'ㅠㅠㅠ', 'ㅜ', 'ㅜㅜ', 'ㅜㅜㅜ',
            'ㅡ', '.', '..', '...', '?', '??', '???', '!', '!!',
            'ㅇ', 'ㅇㅇ', 'ㄴ', 'ㄴㄴ', 'ㄱ', 'ㄷ', 'ㅁ',
            '0', 'v', '0v0', '-', '_', 'ㅡㅡ', 'ㅡ.ㅡ',
            '^', '^^', '^^^', ';', ';;', ';;;', '~', '~~',
            '*', '**', '@', '#', '&', '+',
            '(', ')', '[', ']', '<', '>',
            'ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ', 'ㅣ', 'ㅐ', 'ㅔ', 'ㅑ', 'ㅕ',
            '아', '어', '오', '우', '음', '응', '엉', '앙', '잉',
            '헉', '헐', '흠', '흡', '훕', '허', '하', '히', '호', '후',
            '에', '애', '야', '와', '워', '웅', '읭',
            'ㄱㄱ', 'ㄱㅅ', 'ㅊㅋ', 'ㅇㅋ', 'ㄴㄱ',
            'ㅎㅇ', 'ㅂㅂ', 'ㅂㅇ', 'ㄷㄷ', 'ㄱㄷ',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '은', '는', '이', '가', '을', '를', '의', '에', '로', '으로',
            '고', '며', '면', '도', '만', '요', '네', '죠', '지',
            '다', '니', '까', '서', '라', '나', '데',
            '그', '저', '것', '거', '수', '때', '더', '안', '못',
            '좀', '잘', '왜', '뭐', '어디', '누구', '언제', '어떻게',
            '진짜', '정말', '너무', '아주', '매우', '완전',
            '그냥', '일단', '근데', '그래서', '하지만', '그런데',
        }

        harmful_tokens = {}
        for token, value in zip(cleaned_tokens, values):
            if value > 0 and token and len(token) > 0:
                if token not in EXCLUDE_TOKENS:
                    harmful_tokens[token] = float(value)
        
        harmful_tokens = dict(sorted(harmful_tokens.items(), key=lambda x: x[1], reverse=True))
        
        return {"harmful_tokens": harmful_tokens}
        
    except Exception as e:
        print(f"[WARN] SHAP 분석 실패: {e}")
        return {"harmful_tokens": {}}


def predict(text: str) -> dict:
    """단일 텍스트 유해표현 검출"""
    load_model()
    
    if not text or not text.strip():
        return {
            "text": text,
            "percentage": 0.0,
            "is_harmful": False,
            "label": "비유해",
            "bad_words": [],
        }
    
    # 모델 분석용 텍스트 (ㅋㅎㅠㅜㅡ 등 제거)
    text_for_model = re.sub(r'[ㅋㅎㅠㅜㅡ]+', '', text)
    
    # 제거 후 내용이 거의 없으면 통과
    if len(text_for_model.strip()) < 2:
        return {
            "text": text,
            "percentage": 0.0,
            "is_harmful": False,
            "label": "비유해",
            "bad_words": [],
        }
    
    inputs = tokenizer(
        text_for_model,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze())
        
        if len(probs.shape) == 0:
            harmful_prob = probs.item()
        else:
            harmful_prob = probs[1].item()
        
        percentage = round(harmful_prob * 100, 1)
        is_harmful = percentage >= 80.0
        label = "유해표현포함" if is_harmful else "비유해"
    
    bad_words = []
    
    if is_harmful and SHAP_AVAILABLE:
        token_analysis = extract_harmful_tokens(text_for_model)
        bad_words_score = token_analysis.get("harmful_tokens", {})
        bad_words = [word for word, score in bad_words_score.items() if score >= 0.1]
    
    # 초성 욕설 검사
    slang_found = check_slang(text)
    for slang in slang_found:
        if slang not in bad_words:
            bad_words.append(slang)
    
    # 초성 욕설 있으면 유해 판정
    if slang_found and not is_harmful:
        percentage = 95.0
        is_harmful = True
        label = "유해표현포함"
    
    return {
        "text": text,
        "percentage": percentage,
        "is_harmful": is_harmful,
        "label": label,
        "bad_words": bad_words,
    }


def handler(event, context):
    """AWS Lambda 핸들러"""
    try:
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body") or event
        
        if "text" in body:
            text = body["text"]
            result = predict(text)
            
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps(result, ensure_ascii=False)
            }
        
        elif "texts" in body:
            texts = body["texts"]
            results = []
            harmful_count = 0
            
            for text in texts:
                result = predict(text)
                results.append(result)
                if result["is_harmful"]:
                    harmful_count += 1
            
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "results": results,
                    "total": len(results),
                    "harmful_count": harmful_count
                }, ensure_ascii=False)
            }
        
        else:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "text 또는 texts 필드 필요"})
            }
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }