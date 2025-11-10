# 패키지 설치 가이드

## AgentExecutor 오류 해결 방법

`AgentExecutor`가 설치되지 않았다는 오류는 LangChain 패키지가 설치되지 않아서 발생합니다.

## 해결 방법

### 1. requirements.txt를 사용한 설치 (권장)

```bash
pip install -r requirements.txt
```

### 2. 개별 패키지 설치

```bash
pip install langchain langchain-community langchain-core
```

### 3. 전체 패키지 설치

```bash
pip install langchain>=0.1.0 langchain-community>=0.0.20 langchain-core>=0.1.0 openpyxl>=3.1.0 faiss-cpu>=1.7.4 pandas>=2.0.0
```

## 설치 확인

설치가 완료되었는지 확인하려면:

```bash
python3 -c "from langchain.agents import AgentExecutor; print('AgentExecutor 설치 완료!')"
```

또는

```bash
python3 -c "import langchain; print('LangChain 버전:', langchain.__version__)"
```

## 가상환경 사용 (권장)

프로젝트별로 가상환경을 사용하는 것을 권장합니다:

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

## 문제 해결

### 패키지 설치 오류 시

1. **pip 업그레이드**:
   ```bash
   pip install --upgrade pip
   ```

2. **특정 버전 설치**:
   ```bash
   pip install langchain==0.1.0 langchain-community==0.0.20
   ```

3. **Python 버전 확인**:
   - Python 3.8 이상 필요
   - `python3 --version`으로 확인

### 설치 후에도 오류가 발생하는 경우

1. Python 인터프리터 경로 확인
2. IDE(VS Code)에서 올바른 Python 인터프리터 선택
3. 가상환경이 활성화되어 있는지 확인

