import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import numpy as np

print("🔧 Configurando modelo CNN...")

# Cargar modelo pre-entrenado (ResNet50)
model = models.resnet50(pretrained=True)
model.eval()  # Modo evaluación

print("Modelo ResNet50 cargado")

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Mapeo de clases ImageNet a equipos de gym
# ImageNet tiene 1000 clases, estas son las relevantes para gym
EQUIPO_MAPPING = {
    'dumbbell': ['Mancuernas'],
    'barbell': ['Barra'],
    'weight': ['Pesas', 'Discos'],
    'bench': ['Banco'],
    'treadmill': ['Máquinas'],
    'band': ['Bandas elásticas'],
    'rubber_band': ['Bandas elásticas'],
    'gym': ['Máquinas'],
    'fitness': ['Equipamiento variado'],
}

def analizar_imagen(imagen_path):
    """
    Analiza una imagen y detecta qué equipos de gym contiene
    
    Args:
        imagen_path: ruta a la imagen o objeto PIL Image
    
    Returns:
        dict: equipos detectados con confianza
    """
    # Cargar imagen
    if isinstance(imagen_path, str):
        img = Image.open(imagen_path).convert('RGB')
    else:
        img = imagen_path.convert('RGB')
    
    # Aplicar transformaciones
    img_tensor = transform(img).unsqueeze(0)
    
    # Hacer predicción
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Obtener top 5 predicciones
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Cargar labels de ImageNet
    LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    try:
        import urllib.request, json
        with urllib.request.urlopen(LABELS_URL) as url:
            labels = json.loads(url.read().decode())
    except:
        # Backup: labels básicos
        labels = [f"clase_{i}" for i in range(1000)]
    
    # Analizar predicciones
    equipos_detectados = []
    confianzas = {}
    
    for prob, catid in zip(top5_prob, top5_catid):
        label = labels[catid.item()].lower()
        confidence = prob.item()
        
        # Buscar matches con equipos de gym
        for keyword, equipos in EQUIPO_MAPPING.items():
            if keyword in label:
                for equipo in equipos:
                    if equipo not in confianzas or confidence > confianzas[equipo]:
                        confianzas[equipo] = confidence
                        if equipo not in equipos_detectados:
                            equipos_detectados.append(equipo)
    
    # Si no se detectó nada específico, analizar contexto general
    if not equipos_detectados:
        # Buscar palabras relacionadas con fitness en top predictions
        top_labels = [labels[catid.item()].lower() for catid in top5_catid]
        
        if any(word in ' '.join(top_labels) for word in ['weight', 'dumbbell', 'barbell']):
            equipos_detectados.append("Pesas")
            confianzas["Pesas"] = 0.6
        else:
            # Por defecto: solo peso corporal
            equipos_detectados.append("Solo peso corporal")
            confianzas["Solo peso corporal"] = 0.8
    
    return {
        'equipos': equipos_detectados,
        'confianzas': confianzas
    }

def equipos_a_lista(equipos_detectados):
    """
    Convierte equipos detectados al formato exacto del selector de la app
    """
    # Las opciones exactas del multiselect
    opciones_validas = ["Pesas", "Mancuernas", "Barra", "Bandas elásticas", 
                        "Máquinas", "Solo peso corporal"]
    
    mapeo_app = {
        'Mancuernas': 'Mancuernas',
        'Barra': 'Barra',
        'Pesas': 'Pesas',
        'Discos': 'Pesas',
        'Bandas elásticas': 'Bandas elásticas',
        'Banco': 'Pesas',
        'Máquinas': 'Máquinas',
        'Solo peso corporal': 'Solo peso corporal',
        'Equipamiento variado': 'Pesas'
    }
    
    resultado = []
    for equipo in equipos_detectados:
        if equipo in mapeo_app:
            equipo_app = mapeo_app[equipo]
            # Solo agregar si está en las opciones válidas y no está duplicado
            if equipo_app in opciones_validas and equipo_app not in resultado:
                resultado.append(equipo_app)
    
    
    return resultado if resultado else []

# Test básico
if __name__ == "__main__":
    print("Módulo CNN listo para usar")
    print("Esperando imágenes desde la app\n")