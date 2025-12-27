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
        return {"harmful_tokens": {}, "safe_tokens": {}}
    
    try:
        explainer = shap.Explainer(pipe)
        shap_values = explainer([text])
        shap_for_text = shap_values[0, :, 1]
        
        values = np.round(shap_for_text.values, 4)
        tokens = shap_for_text.data
        cleaned_tokens = [re.sub(r'^[Ġ▁]', '', str(t)).strip() for t in tokens]
        
        harmful_tokens = {}
        for token, value in zip(cleaned_tokens, values):
            if value > 0 and token and len(token) > 0:
                if token not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '']:
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
            #"bad_words_score": {}
        }
    
    inputs = tokenizer(
        text,
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
    #bad_words_score = {}
    
    if is_harmful and SHAP_AVAILABLE:
        token_analysis = extract_harmful_tokens(text)
        bad_words_score = token_analysis.get("harmful_tokens", {})
        bad_words = [word for word, score in bad_words_score.items() if score >= 0.1]
    
    return {
        "text": text,
        "percentage": percentage,
        "is_harmful": is_harmful,
        "label": label,
        "bad_words": bad_words,
        #"bad_words_score": bad_words_score
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