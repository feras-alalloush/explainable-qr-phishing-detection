import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="QR visual/text/fusion classification; never opens URLs")
    parser.add_argument("--checkpoint", default="models/selected_three_systems.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer")
    parser.add_argument("--image")
    parser.add_argument("--url", help="Decoded URL text; no network request is made")
    parser.add_argument("--mode", choices=["all", "visual", "text", "fusion"], default="all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    args = parser.parse_args()
    from .model import ModelBundle
    import torch
    torch.set_num_threads(1)
    bundle = ModelBundle(args.checkpoint, args.tokenizer, args.device)
    output = bundle.predict(args.image, args.url, args.mode)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
