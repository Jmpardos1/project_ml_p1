# Proyecto — Clasificación de sentimiento de reseñas (Parte 1: ML clásico)

Aprendizaje de Máquina 2026-20, Universidad de los Andes.

Clasificar reseñas de productos en español en `negativo`, `neutral` o `positivo` usando **solo scikit-learn** y
representaciones clásicas (bag-of-words, TF-IDF, n-gramas). Métrica de Kaggle: **accuracy**.

> Estado al 1 de octubre de 2026: 3 experimentos terminados, 3 envíos generados (pendientes de subir a Kaggle).
> Mejor modelo hasta ahora: **experimento 2** (accuracy CV 0,887 · test 0,899).

---

## 1. Resumen de resultados

| # | Notebook | Idea | Acc. CV | Acc. test | Robustez* | Envío | Kaggle público |
|---|---|---|---|---|---|---|---|
| 1 | `experimento1.ipynb` | Línea base: TF-IDF (1,2) + Regresión logística | 0,736 ± 0,013 | 0,724 | 0,721 | `submission_01.csv` | pendiente |
| 2 | `experimento2.ipynb` | TF-IDF por partes: texto completo + **última oración** + **tras el último conector** | **0,887 ± 0,010** | **0,899** | 0,686 | `submission_02.csv` | pendiente |
| 3 | `experimento3.ipynb` | TF-IDF por partes: texto completo + **oraciones con conector conclusivo** (sin usar posición) | 0,798 ± 0,015 | 0,796 | **0,800** | `submission_03.csv` | pendiente |

\* *Robustez:* accuracy en el test con las oraciones de cada reseña desordenadas al azar (mide cuánto depende el
modelo de que el veredicto esté al final). En los tres experimentos, `C = 10` fue elegido por la CV.

**Para la competencia, el mejor modelo es el del experimento 2.** Cuando suban los envíos, anoten el score público
en esta tabla y en la columna `kaggle_publico` del notebook correspondiente.

---

## 2. Estructura del repositorio

```
project_ml_p1/
├── README.md                 # este documento
├── train.csv                 # 12.000 reseñas con etiqueta
├── eval.csv                  # 3.000 reseñas sin etiqueta (solo para predecir; PROHIBIDO entrenar con ellas)
├── sample_submission.csv     # formato de envío
├── experimento1.ipynb        # exploración + preprocesamiento + validación + experimento 1
├── experimento2.ipynb        # experimento 2 (+ prueba de robustez)
├── experimento3.ipynb        # experimento 3 (veredicto por contenido)
├── envios/
│   └── submission_XX.csv     # un archivo por experimento enviado
└── modelos/
    └── modelo_XX.joblib      # pipeline entrenado con las 12.000 reseñas, uno por envío
```

`experimento1.ipynb` es el notebook "base": contiene la **exploración de datos** y la **justificación del
preprocesamiento**. Los demás notebooks solo repiten un resumen de la configuración.

---

## 3. Entorno y cómo ejecutar

| | Versión usada |
|---|---|
| Python | 3.14.6 (Anaconda) |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| nltk | 3.10.0 (instalado, aún no usado) |

- En VS Code seleccionen el kernel de **Anaconda** (`anaconda3`). El Python del sistema no tiene las librerías.
- Para comprobar resultados: **Restart & Run All**. Ejecutar celdas sueltas o en desorden puede dejar variables
  viejas y dar resultados que no corresponden al notebook.
- Los resultados son **determinísticos**: misma semilla, mismo split y mismos folds. Ejecutar de nuevo un notebook
  completo da exactamente las mismas métricas. Con otras versiones de las librerías podrían cambiar los últimos
  decimales.
- Tiempos aproximados: experimento 1 ≈ 1 min, experimento 2 ≈ 3 min, experimento 3 ≈ 4 min.
- **Cargar un `.joblib`:** los modelos 2 y 3 usan un `FunctionTransformer(separar_partes)`, y `joblib` guarda la
  función **por referencia**. Para cargarlos hay que tener definidas antes las funciones del notebook
  (`dividir_oraciones`, `separar_partes`, etc.) y pasarles texto ya normalizado con `normalizar_texto()`.

---

## 4. Plantilla de los notebooks

Cada experimento va en **su propio notebook autocontenido**: `experimentoN.ipynb`. Estructura:

```
# Experimento N — <nombre corto>
  Intro: qué continúa, aviso de que los experimentos anteriores NO se re-ejecutan, contenido.

## 1. Configuración común (resumen)            ← copiar tal cual de experimento2/3.ipynb
   1.1 Librerías, semilla y datos               (SEED = 42, CLASES, carga con .copy())
   1.2 Normalización base                       (normalizar_texto, aplicada a train y eval)
   1.3 Validación y métricas                    (split 80/20 estratificado, StratifiedKFold(5), SCORING)
   1.4 Funciones auxiliares                     (tabla_cv, evaluar_en_test, registrar_resultado,
                                                 generar_envio, top_coeficientes)
   + celda con REFERENCIA_E1, REFERENCIA_E2, ... (métricas fijas de experimentos anteriores)

## 2. Experimento N — <nombre>
   Motivación (qué problema del experimento anterior ataca) · Hipótesis · Proceso (pasos)
   2.1 ... 2.k   Diagnóstico / comparación de variantes con CV sobre X_train
                 → después de cada tabla o gráfico: celda markdown "Lectura de ..."
   2.x Modelo del experimento N: GridSearchCV con SU PROPIA grilla de C (u otro hiperparámetro)
   2.x Evaluación en test (evaluar_en_test) + coeficientes
   2.x Prueba de robustez (oraciones desordenadas / última oración al inicio)
   2.x Errores que quedan (cross_val_predict sobre X_train, NO sobre el test)
   2.x Registro (registrar_resultado) y envío (generar_envio(busqueda, numero=N))
   2.x Análisis del experimento: resultados, comparación, alcances y límites, próximos pasos
```

### Reglas que seguimos en todos los experimentos

| Regla | Detalle |
|---|---|
| **Nunca entrenar con `eval.csv`** | Solo se usa para predecir el envío. |
| **Mismo split y semilla** | `SEED = 42`; `train_test_split(test_size=0.20, stratify=y, random_state=SEED)`; `StratifiedKFold(5, shuffle=True, random_state=SEED)`. Así todos los experimentos son comparables. |
| **Roles de los datos** | `X_train` (9.600): búsqueda con CV. `X_test` (2.400): hold-out que **solo se reporta**. Leaderboard privado de Kaggle: test definitivo. |
| **Se decide con CV, no con el test** | Elegir configuraciones e ideas con la accuracy media de CV. Mirar el test para decidir es leakage leve. |
| **Análisis de errores sobre train** | Con `cross_val_predict` sobre `X_train`, nunca con los errores del test. |
| **Todo lo que aprende de los datos va dentro del `Pipeline`** | Vocabulario, IDF, etc. Así cada fold solo ve sus datos de entrenamiento. |
| **Cada experimento ajusta su propio `C`** | El óptimo cambia con las features. |
| **Métricas** | Se decide con **accuracy** (`refit="accuracy"`); se reporta **F1 macro** como control. |
| **¿Una mejora es real?** | Debe superar la desviación entre folds; si es dudosa, comparar **fold por fold** (mismos folds). Ante un empate, elegir lo más simple. |
| **No re-ejecutar experimentos anteriores** | Sus métricas se escriben como valores fijos (`REFERENCIA_EN`). Las configuraciones anteriores solo se incluyen como filas de referencia cuando se mide algo nuevo que no existía en su notebook. |
| **Envío** | `generar_envio()` reentrena con las 12.000 reseñas, valida el formato (`id,answer`, 3.000 filas, 3 etiquetas) y guarda el CSV y el `.joblib`. |
| **Documentación** | Todo en español. Cada decisión, supuesto y descarte se justifica en markdown, incluyendo **por qué no es leakage** y **qué supuesto introduce** una feature. |

### Normalización base (fija en todos los experimentos)

`normalizar_texto()`: reparar mojibake (`aquÃ­` → `aquí`) · minúsculas · quitar tildes **conservando la ñ** ·
letras repetidas 3+ → 2 (`errror` → `error`) · colapsar espacios.

**No** se quitan stopwords (las listas en español incluyen `no`, `pero`, `aunque`, `nada`, que son clave aquí) ni la
puntuación (hace falta para separar oraciones).

---

## 5. Conclusiones hasta ahora

### Exploración (`experimento1.ipynb`, secciones 1–2)
- Clases casi balanceadas: 35,1 % / 29,8 % / 35,1 %. Predecir siempre la mayoritaria da 35,1 %.
- La longitud no distingue clases. Train y eval tienen la misma estructura (≈ 3,9 oraciones por reseña).
- `neutral` son reseñas **descriptivas**: casi sin conectores de juicio y con más cifras y siglas (`USB`, `110 voltios`).
- `negativo` y `positivo` usan los mismos conectores con la misma frecuencia: lo que decide es **qué viene después**.
- Hay ruido inyectado: 507 palabras con y sin tilde, el 40 % del vocabulario aparece una sola vez (muchos errores
  de tipeo como `coomprar`), letras repetidas y mojibake.

### Experimento 1 — Línea base
- `neutral` queda prácticamente resuelta (F1 0,99). **Todo el error está entre `negativo` y `positivo`**: solo
  60,7 % de acierto entre esas dos clases, poco más que el azar.
- Sobreajuste fuerte (train ≈ 1,0), pero la CV es plana entre `C = 3` y `C = 100`: **el límite no es la
  regularización, sino que la bolsa de palabras no ve el orden**.

### Experimento 2 — Final de la reseña
- Diagnóstico: entrenando con una sola parte, la primera oración da 0,51, el texto completo 0,74 y la **última
  oración 0,85**. El veredicto está al final.
- Mejor combinación: texto completo + última oración + tras el último conector, cada parte con su propio TF-IDF
  (gana a "texto + última" en los 5 folds).
- Negativo vs. positivo en test: de 60,7 % a **85,9 %**.
- **Prueba de robustez:** con las oraciones desordenadas cae a 0,686 y con la última oración al inicio a 0,574,
  **por debajo de la línea base**. Aprendió una regla de **posición** que funciona porque `eval.csv` tiene la misma
  estructura que train, pero no debería llevarse a reseñas de otra fuente sin validarlo.

### Experimento 3 — Veredicto por contenido
- Idea: identificar el veredicto por **cómo empieza la oración** (`aun asi`, `al final`, `con todo`,
  `a fin de cuentas`, `pero`, `sin embargo`...), no por dónde está.
- Robusto: ≈ 0,80 en todos los escenarios de la prueba de robustez. Pero en los datos originales queda 9 puntos
  por debajo del experimento 2.
- **Techo de información:** solo el 31 % de las reseñas negativo/positivo tiene un conector conclusivo. En el resto,
  el veredicto es la **última opinión sin marca**: "A excelente. B pésima." (negativo) y "B pésima. A excelente."
  (positivo) tienen las mismas palabras. Sin usar el orden no se pueden distinguir.
- Matiz: la prueba de robustez le favorece un poco, porque conserva el conector aunque la oración quede al inicio. En
  un caso realista (veredicto al inicio **sin** conector) el experimento 3 no se invierte, pero rinde como la línea base.
- **N-gramas más largos no ayudan** (texto completo, CV): (1,1) 0,739 · (1,2) 0,736 · (1,3) 0,728 · (1,6) 0,717.
  El orden relevante es de largo alcance (10–30 palabras) y los n-gramas largos son casi únicos.

### Decisión
**Competir con la base del experimento 2** y reportar la robustez en cada experimento siguiente. Los experimentos 2 y
3 quedan como evidencia del compromiso entre accuracy y robustez para la sustentación, y motivan la Parte 2 (modelos
secuenciales).

### Descartes documentados
| Descartado | Motivo |
|---|---|
| KNN | Alta dimensión: las distancias se concentran; no aprende pesos con la etiqueta (TF-IDF pondera por rareza, no por relevancia). |
| Random Forest | Cada predicción consulta ≈ 14 features y el sentimiento es una suma de muchas pistas débiles; muestrear features en datos dispersos rinde mal. |
| SGDClassifier | Mismo modelo lineal con un optimizador aproximado; no aporta frente a LR y LinearSVC con 12.000 reseñas. |
| MLPClassifier | Es una red neuronal: **prohibida** en la Parte 1. |
| Lematización con spaCy, embeddings | Los modelos de spaCy y los embeddings son redes neuronales: prohibidos. |
| Quitar stopwords con listas estándar | Eliminan `no`, `pero`, `aunque`, `nada`. |
| L1 como regularización por defecto | Se queda con una feature por grupo de correlacionadas; en texto rinde igual o peor que L2. Se probará como variante. |

---

## 6. Próximos experimentos (sobre la base del experimento 2)

Máximo ≈ 10 envíos en total. Se envía solo si la mejora en CV supera el ruido entre folds. Todos reportan también la
prueba de robustez.

| # | Experimento | Qué probar | Por qué |
|---|---|---|---|
| 4 | **Última oración con opinión + tipo de conector** | Saltar oraciones finales de relleno (`ni lo mejor ni lo peor`, `ahí queda mi experiencia`, `todavía no me decido`); marcar el tipo de conector (conclusivo / concesivo / `eso sí`) | Es el patrón de error más frecuente que queda en el experimento 2 (16 % de error negativo/positivo) |
| 5 | **Representación** | `min_df`, `max_df`, n-gramas por parte, BoW frente a TF-IDF, `sublinear_tf`; normalización: stemming (Snowball), números → token, corrección de tipeo por distancia de edición | Ajuste fino; la ganadora se incorpora a la normalización base |
| 6 | **Matriz representación × modelo** | Mejores 2–3 representaciones × {LR multinomial, LR OvR, LinearSVC, MultinomialNB, ComplementNB} | Las representaciones y los modelos interactúan |
| 7 | **N-gramas de caracteres** | `char_wb` (2,5)/(3,5) por parte, solos o con palabras | Robustez a errores de tipeo (`coomprar` ≈ `comprar`) |
| 8 | **Negación** | Marcar `no funciona` → `no_funciona` (`no`, `nunca`, `ni`, `sin`) | `no` y `nada` son más frecuentes en `neutral`; el contexto importa |
| 9 | **Ajuste fino conjunto** | Grilla fina alrededor del mejor par; L1 / Elastic Net (`LogisticRegression(solver="saga")`) | Últimos décimos |
| 10 | Reserva | Solo si el análisis de errores sugiere algo concreto (ensamble por votación, nuevos conectores) | — |

---

## 7. Pendientes

- [ ] **Subir a Kaggle** `submission_01/02/03.csv` y registrar el score público (tabla de la sección 1 y columna
      `kaggle_publico`). Mínimo **5 envíos distintos** para que cuente la participación.
- [ ] Agregar en el análisis del experimento 3 el matiz sobre la prueba de robustez y la tabla de n-gramas largos.
- [ ] Agregar en cada notebook una celda que imprima las versiones de las librerías, y un `requirements.txt`.
- [ ] Al terminar: **consolidar todo en un único notebook final** (la entrega exige uno solo), con una tabla resumen de
      todos los experimentos, y guardar el `.joblib` del **mejor envío del leaderboard público**.
- [ ] Antes de la entrega: meter la normalización dentro del pipeline del modelo final, para que el `.joblib` reciba
      texto crudo.

### Fechas
- Cierre de la competencia de la Parte 1: **semana 9**.
- Entrega en Bloque Neón (notebook + modelo `.joblib`/`pickle` del mejor envío): **semana 11**.
