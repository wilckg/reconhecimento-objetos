# 🎯 Detecção de Objetos em Tempo Real com YOLOv8

Este repositório contém um script em Python para **detecção de objetos** utilizando o modelo **YOLOv8**, da biblioteca [Ultralytics](https://docs.ultralytics.com/).  
Com ele, você pode rodar inferência em:

- 📷 **Webcam**
- 🖼️ **Imagens**
- 🎥 **Vídeos**

É um ótimo ponto de partida para projetos de **Visão Computacional**, **Análise Esportiva**, **Segurança**, ou simplesmente para experimentar redes neurais convolucionais aplicadas à detecção de objetos.

---

## 🧠 Visão geral

O script `yolov8_detect.py`:

- Carrega um modelo YOLOv8 (por padrão, `yolov8n.pt`)
- Aceita diferentes fontes de entrada (`--source`)
- Exibe os resultados em tempo real com bounding boxes
- Opcionalmente salva as saídas em `runs/detect/`
- Permite configurar confiança, IoU e dispositivo (CPU / GPU)

---

## 📦 Pré-requisitos

- Python **3.8+**
- `pip` atualizado
- (Opcional, mas recomendado) GPU NVIDIA com drivers + CUDA configurados

### 🔧 Criando ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
# Linux / WSL
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 📥 Instalando dependências

```bash
pip install --upgrade pip
pip install ultralytics opencv-python
```

---

## 📁 Estrutura do projeto (sugestão)

```bash
.
├── yolov8_detect.py    # Script principal de detecção
├── README.md           # Este arquivo
└── media/              # (Opcional) Imagens e vídeos de teste
    ├── image.jpg
    └── video.mp4
```

---

## 🧾 Script principal (`yolov8_detect.py`)

> 🔎 *Esse é o script esperado pelo README. Caso o seu esteja diferente, é só ajustar aqui depois.*

```python
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
            " - caminho de imagem (ex: media/imagem.jpg)\n"
            " - caminho de vídeo (ex: media/video.mp4)\n"
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
```

---

## ▶️ Como rodar

### 1️⃣ Webcam (padrão)

```bash
python yolov8_detect.py --source 0
```

### 2️⃣ Imagem

```bash
python yolov8_detect.py --source media/imagem.jpg
```

### 3️⃣ Vídeo

```bash
python yolov8_detect.py --source media/video.mp4
```

### 4️⃣ Salvar os resultados (imagem/vídeo com boxes desenhados)

```bash
python yolov8_detect.py --source media/video.mp4 --save
```

Os arquivos processados serão salvos em algo como:

```text
runs/detect/predict/
```

---

## ⚙️ Parâmetros úteis

| Parâmetro   | Descrição |
|------------|-----------|
| `--source` | Fonte de entrada: `0` (webcam), caminho de imagem, caminho de vídeo |
| `--weights`| Caminho do modelo `.pt` (ex: `yolov8n.pt`, `yolov8s.pt`, modelo treinado) |
| `--conf`   | Confiança mínima das detecções (padrão: `0.5`) |
| `--iou`    | IoU para supressão de caixas (NMS) (padrão: `0.45`) |
| `--device` | Dispositivo: `cpu`, `0`, `1`... (GPU) |
| `--save`   | Se presente, salva os resultados em `runs/detect/` |

### Exemplos:

**Forçar CPU:**

```bash
python yolov8_detect.py --source 0 --device cpu
```

**Usar modelo maior (mais preciso, porém mais pesado):**

```bash
python yolov8_detect.py --source 0 --weights yolov8s.pt
```

---

## 🧪 Melhorando o projeto (idéias futuras)

- Treinar o YOLOv8 com um **dataset específico** (por exemplo, jogadores de futebol)
- Integrar rastreamento com **DeepSORT** ou **ByteTrack**
- Exportar resultados (JSON/CSV) com as detecções por frame
- Criar uma interface web ou dashboard (Streamlit, FastAPI, etc.)

---

## 📚 Referências

- [Ultralytics YOLOv8 – Documentação Oficial](https://docs.ultralytics.com/)
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.*

---

## 👨‍💻 Autor

**Wilck Gomes de Oliveira**  
Projeto acadêmico e exploratório em Visão Computacional e Deep Learning.

Se este repositório foi útil, considere deixar uma ⭐ no GitHub! 🙂
