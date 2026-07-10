# Predicción de Calificación de Clientes a partir de Reseñas de Amazon (NLP a Escala)

Modelo de clasificación que predice la calificación (1-5 estrellas) que un cliente dará a un producto, a partir únicamente del texto libre de su reseña. Entrenado sobre más de 550,000 reseñas reales de la categoría Home & Kitchen de Amazon.

## Por qué importa

En cualquier negocio con reseñas o feedback de clientes (e-commerce, retail, apps, marketplaces), poder anticipar el sentimiento de un cliente a partir de texto no estructurado permite:
- Detectar productos problemáticos antes de que el rating baje visiblemente
- Priorizar respuesta a reseñas negativas de forma automática
- Monitorear satisfacción a escala sin revisar cada reseña manualmente

## Dataset

- **Fuente:** [Amazon Product Reviews Dataset](http://jmcauley.ucsd.edu/data/amazon/) — categoría Home & Kitchen
- **Tamaño:** 551,682 reseñas, 28,237 productos únicos (ASIN)
- **Columnas clave:** texto de reseña (`reviewText`), calificación (`overall`), resumen (`summary`), utilidad percibida (`helpful`)

## Metodología

1. **Análisis exploratorio:** distribución de calificaciones, cantidad de reseñas por producto, análisis de la escala de utilidad (`helpful`)
2. **Preprocesamiento de texto:** limpieza y vectorización del texto de las reseñas (resultado guardado en `review_text_preprocesado.pkl` para evitar reprocesar en cada ejecución)
3. **Modelado:** clasificación multiclase (1 a 5 estrellas) con dos algoritmos:
   - Regresión Logística
   - Random Forest
4. **Validación:** cross-validation de 5 folds + evaluación en conjunto de test separado
5. **Comparación de escala:** se entrenó primero con una muestra de 15,000 reseñas y luego con el dataset completo (550k), para medir el impacto real del volumen de datos en el desempeño del modelo

## Resultados

| Modelo | Accuracy (test) |
|---|---|
| Regresión Logística | 63% |
| Random Forest | 60% |

**Hallazgo clave:** ambos modelos son fuertes clasificando reseñas de 5 estrellas (F1-score de 0.77-0.80), pero pierden precisión notablemente en reseñas de 2 estrellas (F1-score ~0.16-0.26). Esto refleja un desbalance de clases típico en datos reales de reseñas (los clientes satisfechos escriben más reseñas que los moderadamente insatisfechos), y es un punto de mejora identificado para una siguiente iteración (técnicas de balanceo de clases, modelos más robustos como XGBoost o embeddings con transformers).

Al comparar contra el entrenamiento con solo 15,000 reseñas, el modelo entrenado con el dataset completo (550k) mostró mejoras consistentes en accuracy, confirmando el valor de trabajar a escala real en lugar de muestras reducidas.

## Stack técnico

- Python (pandas, scikit-learn)
- Procesamiento de lenguaje natural (NLP) para texto no estructurado
- Google Colab (entrenamiento con GPU)

## Próximos pasos

- Balanceo de clases (SMOTE u oversampling) para mejorar recall en reseñas negativas
- Probar embeddings pre-entrenados (BERT/transformers) en lugar de vectorización clásica
- Desplegar como API simple para scoring de reseñas en tiempo real

---
**Autor:** Santiago Cisneros — [github.com/Shantiago27](https://github.com/Shantiago27)
