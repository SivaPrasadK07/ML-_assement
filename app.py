
#FASTAPI ENDPOINT FOR RESEARCH CENTER QUALITY CLASSIFICATION


import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


#App initialisation


app = FastAPI(
    title="Research Center Quality Classifier",
    description=(
        "Classifies UK research centers into Premium, Standard, or Basic "
        "quality tiers using a trained K-Means clustering model."
    ),
    version="1.0.0",
)


#Load model artifacts


MODEL_PATH = "cluster_model.pkl"

def _load_or_train():
    """Load saved artifacts, or train from scratch if not yet saved."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    #train the model so the API works standalone
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    df = pd.read_csv("research_centers.csv")

    selected_features = [
        "internalFacilitiesCount",
        "hospitals_10km",
        "pharmacies_10km",
        "facilityDiversity_10km",
        "facilityDensity_10km",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Map cluster indices to quality tiers based on mean internalFacilitiesCount
    df["cluster"] = kmeans.labels_
    cluster_means = df.groupby("cluster")["internalFacilitiesCount"].mean().sort_values(ascending=False)
    tiers = ["Premium", "Standard", "Basic"]
    cluster_to_tier = {int(cluster): tier for cluster, tier in zip(cluster_means.index, tiers)}

    artifacts = (kmeans, scaler, selected_features, cluster_to_tier)
    joblib.dump(artifacts, MODEL_PATH)
    return artifacts


kmeans, scaler, selected_features, cluster_to_tier = _load_or_train()


#Request / Response schemas


class ResearchCenterInput(BaseModel):
    internalFacilitiesCount: float = Field(..., ge=0, description="Number of internal facilities")
    hospitals_10km: float = Field(..., ge=0, description="Number of hospitals within 10 km")
    pharmacies_10km: float = Field(..., ge=0, description="Number of pharmacies within 10 km")
    facilityDiversity_10km: float = Field(..., ge=0, le=1, description="Diversity index (0–1)")
    facilityDensity_10km: float = Field(..., ge=0, description="Facility density per km²")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "internalFacilitiesCount": 9,
                    "hospitals_10km": 3,
                    "pharmacies_10km": 2,
                    "facilityDiversity_10km": 0.82,
                    "facilityDensity_10km": 0.45,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predictedCluster: int
    predictedCategory: str



#Endpoints


@app.get("/", summary="Health check")
def root():
    """Returns a simple health-check response."""
    return {"status": "ok", "message": "Research Center Quality Classifier is running."}


@app.post("/predict", response_model=PredictionResponse, summary="Classify a research center")
def predict_quality(data: ResearchCenterInput):
    """
    Accepts feature values for a research center and returns its predicted
    quality tier (**Premium**, **Standard**, or **Basic**).
    """
    try:
        input_df = pd.DataFrame([data.model_dump()])[selected_features]
        X_scaled = scaler.transform(input_df)
        cluster_label = int(kmeans.predict(X_scaled)[0])
        tier = cluster_to_tier.get(cluster_label, "Unknown")
        return PredictionResponse(predictedCluster=cluster_label, predictedCategory=tier)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
