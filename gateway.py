"""
Ollama API Gateway/Proxy 서버
API Key 인증을 통해 Ollama API에 접근할 수 있도록 프록시 역할을 수행합니다.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import json
from typing import Optional

# 환경 변수에서 설정 읽기
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
API_KEY = os.getenv("OLLAMA_API_KEY", "my-very-secret-key-123")

app = FastAPI(title="Ollama API Gateway", version="1.0.0")

# CORS 설정 (필요한 경우)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_api_key(request: Request) -> str:
    """
    API Key 검증
    
    Args:
        request: FastAPI Request 객체
    
    Returns:
        str: 검증된 API Key
    
    Raises:
        HTTPException: API Key가 없거나 유효하지 않은 경우
    """
    # Authorization 헤더에서 Bearer 토큰 확인
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        if token == API_KEY:
            return token
    
    # X-API-Key 헤더 확인 (대체 방식)
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key == API_KEY:
        return api_key
    
    # 쿼리 파라미터에서 확인 (선택사항, 보안상 권장하지 않음)
    api_key_query = request.query_params.get("api_key")
    if api_key_query and api_key_query == API_KEY:
        return api_key_query
    
    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key. Use 'Authorization: Bearer <API_KEY>' or 'X-API-Key: <API_KEY>' header."
    )


@app.post("/api/chat")
async def chat(request: Request):
    """Ollama /api/chat 엔드포인트 프록시"""
    check_api_key(request)
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    async with httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=None) as client:
        try:
            resp = await client.post("/api/chat", json=body, timeout=None)
            resp.raise_for_status()
            
            # 스트리밍 응답인 경우
            if "stream" in body and body["stream"]:
                async def stream_generator():
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="application/json"
                )
            
            # 일반 JSON 응답
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json()
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.post("/api/generate")
async def generate(request: Request):
    """Ollama /api/generate 엔드포인트 프록시"""
    check_api_key(request)
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    async with httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=None) as client:
        try:
            resp = await client.post("/api/generate", json=body, timeout=None)
            resp.raise_for_status()
            
            # 스트리밍 응답인 경우
            if "stream" in body and body.get("stream"):
                async def stream_generator():
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="application/json"
                )
            
            # 일반 JSON 응답
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json()
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.post("/api/embeddings")
async def embeddings(request: Request):
    """Ollama /api/embeddings 엔드포인트 프록시"""
    check_api_key(request)
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    async with httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=None) as client:
        try:
            resp = await client.post("/api/embeddings", json=body, timeout=None)
            resp.raise_for_status()
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json()
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.post("/api/embed")
async def embed(request: Request):
    """Ollama /api/embed 엔드포인트 프록시 (OllamaEmbeddings 호환)"""
    check_api_key(request)
    
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    async with httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=None) as client:
        try:
            # /api/embed를 /api/embeddings로 리다이렉트
            resp = await client.post("/api/embeddings", json=body, timeout=None)
            resp.raise_for_status()
            return JSONResponse(
                status_code=resp.status_code,
                content=resp.json()
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.get("/health")
async def health():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "service": "Ollama API Gateway"}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("GATEWAY_PORT", 8000))
    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    
    print(f"Ollama API Gateway 시작 중...")
    print(f"Ollama URL: {OLLAMA_API_URL}")
    print(f"API Key 설정: {'설정됨' if API_KEY else '설정되지 않음'}")
    print(f"서버 주소: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port)

