# Ollama API Gateway 사용 가이드

## 개요

Ollama API에 인증을 추가하기 위한 프록시 서버입니다. API Gateway를 통해 Ollama API에 접근할 수 있습니다.

## 아키텍처

```
[Client] ──HTTPS──> [API Gateway/Proxy] ──HTTP──> [Ollama (localhost:11434)]
                       ▲
                       │
              API Key 검증 (헤더/쿼리)
```

## 설치

```bash
pip install fastapi uvicorn httpx
```

## 설정

### 1. 환경 변수 설정

```bash
# API Key 설정
export OLLAMA_API_KEY="my-very-secret-key-123"

# Ollama 서버 URL (기본값: http://localhost:11434)
export OLLAMA_API_URL="http://localhost:11434"

# Gateway 포트 (기본값: 8000)
export GATEWAY_PORT=8000
export GATEWAY_HOST="0.0.0.0"
```

### 2. Gateway 서버 실행

```bash
# 방법 1: Python으로 직접 실행
python gateway.py

# 방법 2: uvicorn으로 실행
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### 1. `/api/chat` - 채팅 API

```bash
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-very-secret-key-123" \
  -d '{
    "model": "llama3.1",
    "messages": [
      {"role": "user", "content": "안녕?"}
    ]
  }'
```

### 2. `/api/generate` - 생성 API

```bash
curl http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-very-secret-key-123" \
  -d '{
    "model": "llama3.1",
    "prompt": "안녕하세요"
  }'
```

### 3. `/api/embeddings` - 임베딩 API

```bash
curl http://localhost:8000/api/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-very-secret-key-123" \
  -d '{
    "model": "mxbai-embed-large",
    "prompt": "임베딩할 텍스트"
  }'
```

### 4. `/health` - 헬스 체크

```bash
curl http://localhost:8000/health
```

## 클라이언트 사용법

### Python 코드에서 사용

```python
from ollama_client import LLMAPIClient
import os

# API Key 설정
os.environ["OLLAMA_API_KEY"] = "my-very-secret-key-123"

# LLMAPIClient 초기화
client = LLMAPIClient(
    api_endpoint="http://localhost:8000/api/chat",
    base_url="http://localhost:8000",
    chat_model="llama3.1",
    api_key=os.getenv("OLLAMA_API_KEY")
)
```

## 인증 방식

Gateway는 다음 방식으로 API Key를 확인합니다:

1. **Authorization 헤더** (권장): `Authorization: Bearer <API_KEY>`
2. **X-API-Key 헤더**: `X-API-Key: <API_KEY>`
3. **쿼리 파라미터** (보안상 권장하지 않음): `?api_key=<API_KEY>`

## 보안 권장사항

1. **HTTPS 사용**: 프로덕션 환경에서는 HTTPS를 사용하세요.
2. **강력한 API Key**: 복잡한 API Key를 사용하세요.
3. **환경 변수 사용**: API Key를 코드에 하드코딩하지 마세요.
4. **쿼리 파라미터 사용 금지**: API Key를 URL에 포함하지 마세요.

