#!/usr/bin/env python
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detecção de objetos com YOLOv8")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Fonte de entrada:\n"
            " - caminho de imagem (ex: data/img.jpg)\n"
            " - caminho de vídeo (ex: data/video.mp4)\n"
            " - webcam (use '0' para webcam padrão)"
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Caminho para o modelo YOLOv8 (.pt). Ex: yolov8n.pt, yolov8s.pt, modelo_treinado.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confiança mínima para exibir detecções (0.0 a 1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU para NMS (Non-Max Suppression)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Dispositivo: '' (auto), 'cpu' ou '0', '1' para GPU específica",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Se definido, salva o vídeo/imagens com detecções em runs/detect/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Carrega modelo YOLOv8
    print(f"Carregando modelo: {args.weights}")
    model = YOLO(args.weights)

    # Prepara a fonte: se for "0", trata como webcam
    source = 0 if args.source == "0" else args.source

    print(f"Inferindo em: {source}")
    print("Pressione 'q' na janela de vídeo para encerrar (quando show=True).")

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        show=True,          # mostra janela com as detecções
        save=args.save,     # salva resultados em runs/detect
        stream=False,       # True para stream (processar frame a frame)
        verbose=True,
    )

    # Opcional: imprimir resumo das detecções
    for i, r in enumerate(results):
        if hasattr(r, "boxes") and r.boxes is not None:
            print(f"\n[Frame/Imagem {i}] {len(r.boxes)} objetos detectados:")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                print(f" - {label} ({conf:.2f})")

    print("\n✅ Finalizado.")
    if args.save:
        print("Arquivos salvos em pasta runs/detect/")


if __name__ == "__main__":
    main()
