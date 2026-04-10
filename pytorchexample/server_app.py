"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# 1. ServerApp 생성
app = ServerApp()

# IID
'''
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})    
'''
# Non-IID
@app.main()
def main(grid: Grid, context: Context) -> None:
    """ServerApp의 메인 실행 루프입니다."""
    #Client 수 동적으로 읽어오기
    num_clients = context.run_config.get("num-clients", 3)

    # [수정] 설정값을 안전하게 읽어옵니다. (KeyError 방지)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    lr = context.run_config.get("learning-rate", 0.01)
    
    # rounds 수 설정 (고정)
    num_rounds = 10

    # 전역 모델 초기화
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # 2. [수정] FedAvg 전략 설정
    # 여긴 보통 잘 안건든다 Quickstart 버전이라
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # 3. 학습 시작
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate, 
    )

    # 결과 저장
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """서버에서 전역 모델의 성능을 직접 측정합니다."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 테스트 데이터 로드 및 평가
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    # 서버 로그에 이 수치가 찍히게 됩니다.
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})