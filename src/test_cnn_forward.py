import torch

from models.cnn import CNN2DClassifier, count_trainable_parameters


def main() -> None:
    batch_size = 16
    num_antennas = 3
    num_subcarriers = 30
    window_size = 30
    num_classes = 176

    model = CNN2DClassifier(
        input_channels=num_antennas,
        num_classes=num_classes,
        dropout_rate=0.3,
    )

    dummy_input = torch.randn(
        batch_size,
        num_antennas,
        num_subcarriers,
        window_size,
    )

    logits = model(dummy_input)

    expected_shape = (batch_size, num_classes)

    print("CNN FORWARD SANITY CHECK")
    print(f"dummy_input shape: {dummy_input.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"expected logits shape: {expected_shape}")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")

    if logits.shape != expected_shape:
        raise ValueError(
            f"Invalid logits shape: {logits.shape}, expected {expected_shape}"
        )

    print()
    print("CNN FORWARD PASS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()