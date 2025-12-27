# 에고북 유해표현 검출 API

에고북 서비스에서 편지, 답장 텍스트에서 유해 표현을 탐지하는 AI API입니다.

---

## ⚙️ 기술 스택

TTA(한국정보통신기술협회)에서 개발한 '유해표현 검출 AI모델'을 활용하였습니다.
- **모델**: KcElectra (F1-Score: 0.9928)
- **인프라**: AWS Lambda + API Gateway
- **분석**: SHAP (유해 단어 추출)

---

## 📡 API 정보

### Request
```json
{
  "text": "검사할 텍스트"
}
```

### Response
```json
{
  "text": "검사할 텍스트",
  "percentage": 100.0,
  "is_harmful": true,
  "label": "유해표현포함",
  "bad_words": ["xx", "xx"]
}
```

### 응답 필드 설명

| 필드 | 타입 | 설명 |
|-----|------|------|
| `text` | String | 입력한 텍스트 |
| `percentage` | Float | 유해 확률 (0~100) |
| `is_harmful` | Boolean | 유해 여부 (80% 이상이면 true) |
| `label` | String | "유해표현포함" 또는 "비유해" |
| `bad_words` | List | 유해 판정에 영향을 준 단어들 |

---

## 🎯 판단 기준

- `is_harmful == true` → 전송 차단
- `is_harmful == false` → 전송 허용

**80% 이상**이면 유해로 판정합니다.


---

## 🧪 테스트 예시

### 유해 텍스트
```bash
curl -X POST "https://-/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "xx같은 쓰레기야"}'
```

**응답:**
```json
{
  "text": "xx같은 xx야",
  "percentage": 100.0,
  "is_harmful": true,
  "label": "유해표현포함",
  "bad_words": ["xx", "xx"]
}
```

### 정상 텍스트
```bash
curl -X POST "https://-/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "오늘 하루도 화이팅!"}'
```

**응답:**
```json
{
  "text": "오늘 하루도 화이팅!",
  "percentage": 0.1,
  "is_harmful": false,
  "label": "비유해",
  "bad_words": []
}
```

---


## 📊 유해 카테고리 (11종)

모델이 학습한 유해 표현 카테고리:

| 카테고리 | 설명 |
|---------|------|
| 욕설 | 일반적인 비속어 |
| 모욕 | 인격 비하 표현 |
| 외설 | 성적 표현 |
| 장애 | 장애인 비하 |
| 인종/지역 | 인종/지역 차별 |
| 연령 | 연령 차별 |
| 종교 | 종교 비하 |
| 정치성향 | 정치적 비하 |
| 직업 | 직업 비하 |
| 성혐오 | 성별 혐오 |
| 폭력위협/범죄조장 | 위협/범죄 관련 |

---

## ⚠️ 주의사항

1. **첫 요청 지연**: 콜드 스타트 시 3~5분 소요될 수 있음 (워밍 스케줄러 14분)
2. **UTF-8 인코딩**: 요청 시 반드시 UTF-8로 전송
3. **최대 텍스트 길이**: 512 토큰 (약 1000자 내외, 실제 서비스는 360자 이하로 제한)