---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

## ## 🚀 GitHub 개발 워크플로우 가이드

### 1. 프로젝트 초기 설정 (최초 1회)
이 과정은 프로젝트 폴더를 GitHub 저장소와 처음 연결할 때만 수행합니다.

```bash
# 1. Git 저장소 초기화
git init

# 2. 브랜치 이름을 main으로 변경 (표준)
git branch -M main

# 3. 원격 저장소 주소 등록
git remote add origin [https://github.com/사용자아이디/리포지토리이름.git](https://github.com/사용자아이디/리포지토리이름.git)

## 만약 수정하고싶다면
# 1. 변경된 파일 확인 (빨간색으로 표시된 파일들)
git status

# 2. 변경된 모든 파일을 업로드 대기열에 추가
git add .

# 3. 변경 사항에 대한 메시지 작성 (기록용)
git commit -m "feat: 어떤 기능을 수정했는지 간단히 작성"

# 4. GitHub 온라인 저장소로 전송
git push origin main

---

### ## 💡 백엔드 개발자를 위한 추가 조언

**1. 커밋 메시지 규칙 (Convention)**
나중에 본인이 쓴 메시지를 보고 "이때 뭘 고쳤더라?" 하고 헷갈리지 않으려면 일정한 규칙을 정해두는 게 좋습니다.
* `feat:` : 새로운 기능 추가
* `fix:` : 버그 수정
* `docs:` : README 같은 문서 수정
* `refactor:` : 코드 구조 개선

**2. VS Code UI 활용하기**
터미널 명령어에 익숙해지는 것이 가장 좋지만, VS Code 왼쪽의 **'소스 제어(Source Control)'** 아이콘(세 개의 점과 선 모양)을 클릭하면 마우스 클릭 몇 번으로도 `add`, `commit`, `push`를 할 수 있습니다. 명령어와 매칭하면서 익히면 훨씬 빨라집니다.

이제 이 내용을 `README.md`에 넣고 저장해 보세요! 나중에 다른 프로젝트를 시작할 때 이 파일만 열어보면 바로 명령어를 기억해낼 수 있을 거예요.

성공적으로 README를 작성하셨나요? 혹시 마크다운(Markdown) 문법 중에 더 궁금한 표현 방식이 있으신가요? (예: 표 만들기, 이미지 넣기 등)

# Federated Learning with PyTorch and Flower (Quickstart Example)

This introductory example to Flower uses PyTorch, but deep knowledge of PyTorch is not necessarily required to run the example. However, it will help you understand how to adapt Flower to your use case. Running this example in itself is quite easy. This example uses [Flower Datasets](https://flower.ai/docs/datasets/) to download, partition and preprocess the CIFAR-10 dataset.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

This will create a new directory called `quickstart-pytorch` with the following structure:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in your Flower Configuration. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

> [!TIP]
> For a more detailed walk-through check our [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
