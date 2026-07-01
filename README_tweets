# Clasificación de Tweets Médicos con NLP

Modelo de clasificación binaria de tweets en español sobre COVID-19, entrenado para distinguir entre menciones de **profesionales sanitarios** y menciones de la **población general**. Proyecto desarrollado como parte del Máster en Data Science, Big Data & Business Analytics en la Universidad Complutense de Madrid.

[![Open In Colab](https://colab.research.google.com/drive/1utmnP9W_7jXRHNeMqPyu-dI8gY0oAobL?usp=sharing)

---

## Resultados

| Modelo | F1-Score |
|--------|----------|
| Baseline TF-IDF + Regresión Logística | 0.526 |
| **BETO fine-tuned (3 épocas)** | **0.804** |

Mejora del **+53%** sobre el baseline.

---

## Descripción del problema

El dataset utilizado es **ProfNER** — un corpus de tweets en español anotados manualmente en el contexto de la pandemia COVID-19. La tarea consiste en clasificar cada tweet en una de dos categorías:

- `1` — El tweet menciona a un profesional sanitario
- `0` — El tweet no menciona a un profesional sanitario

La dificultad del problema radica en la naturaleza informal del lenguaje en Twitter: abreviaciones, errores ortográficos, mezcla de idiomas y ambigüedad semántica.

---

## Metodología

### 1. Preprocesamiento
- Limpieza de texto: eliminación de URLs, menciones y caracteres especiales
- Tokenización con el tokenizador nativo de BETO
- Truncado y padding a longitud máxima de 128 tokens

### 2. Baseline
- Vectorización TF-IDF
- Clasificador: Regresión Logística
- F1-Score: **0.526**

### 3. Modelo fine-tuned
- Modelo base: [`dccuchile/bert-base-spanish-wwm-uncased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased) (BETO)
- Fine-tuning supervisado sobre el dataset ProfNER
- 3 épocas de entrenamiento
- F1-Score final: **0.804**

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

- **Transformers:** HuggingFace `transformers` + `datasets`
- **Modelo base:** BETO (`dccuchile/bert-base-spanish-wwm-uncased`)
- **Baseline:** Scikit-learn (TF-IDF + Logistic Regression)
- **Entorno:** Google Colab (GPU T4)

---

## Estructura del repositorio

```
tweet-classification-beto/
│
├── README.md
├── notebook.ipynb        # Notebook completo con código y resultados
└── requirements.txt      # Dependencias del proyecto
```

---

## Cómo ejecutar

1. Abre el notebook en Google Colab con el botón de arriba
2. Ejecuta las celdas en orden
3. El entrenamiento requiere GPU — actívala en `Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU`

---

## Referencias

- Dataset: [ProfNER — NLP4COVID shared task](https://temu.bsc.es/profner/)
- Modelo base: [BETO — dccuchile/bert-base-spanish-wwm-uncased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased)

---

## Autor

**Daniel Molina Novoa** · Data Analyst  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/daniel-molina-novoa-78106925a/)
[![GitHub](https://img.shields.io/badge/GitHub-danielmolinan-181717?style=flat&logo=github&logoColor=white)](https://github.com/danielmolinan)
