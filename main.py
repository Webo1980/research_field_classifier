"""
Research Field Classifier API
=============================
FastAPI application with multiple classification approaches.

Endpoints:
- /classify/{approach} - Classify using specific approach
- /classify/all - Classify using all approaches in parallel
- /health - Health check
"""

import logging
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import uuid
import asyncio

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from services import get_classifier, get_all_classifiers, APPROACHES

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global classifiers
classifiers: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global classifiers
    
    logger.info("Initializing Research Field Classifier API...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Available approaches: {list(APPROACHES.keys())}")
    
    # Initialize all classifiers
    classifiers = get_all_classifiers()
    
    init_tasks = []
    for name, clf in classifiers.items():
        logger.info(f"Initializing {name} classifier...")
        init_tasks.append(clf.initialize())
    
    await asyncio.gather(*init_tasks)
    
    logger.info("All classifiers initialized!")
    
    yield
    
    # Cleanup
    for name, clf in classifiers.items():
        await clf.close()
    
    logger.info("Classifiers shut down")


# FastAPI app
app = FastAPI(
    title="Research Field Classifier API",
    description="""
    Classifies research papers into ORKG research fields using multiple approaches.
    
    ## Approaches
    
    | Approach | Description | Pros | Cons |
    |----------|-------------|------|------|
    | **single_shot** | Full taxonomy to LLM | Global view, no error propagation | Large prompts |
    | **top_down** | Hierarchical navigation | Smaller prompts, structured | Error propagation |
    | **embedding** | LLM + embedding matching | Semantic matching | Two-step process |
    
    ## Usage
    
    1. **Single approach**: `POST /classify/{approach}`
    2. **All approaches**: `POST /classify/all`
    3. **Compare results**: `POST /classify/compare`
    """,
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Request/Response Models ====================

class ClassificationRequest(BaseModel):
    """Request model for classification."""
    raw_input: str = Field(..., description="Paper abstract", min_length=10)
    top_n: int = Field(default=5, ge=1, le=10, description="Number of results")


class AnnotationResult(BaseModel):
    """Single classification result."""
    research_field: str
    research_field_id: str
    score: float
    reasoning: str
    path: List[str]
    path_ids: List[str]


class TimingInfo(BaseModel):
    """Timing information."""
    total_time_ms: float
    llm_time_ms: float
    taxonomy_lookup_ms: float


class TokenUsage(BaseModel):
    """Token usage statistics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ClassificationResponse(BaseModel):
    """Classification response."""
    timestamp: str
    uuid: str
    approach: str
    payload: Dict[str, Any]
    timing: TimingInfo
    token_usage: TokenUsage


class MultiApproachResponse(BaseModel):
    """Response with results from all approaches."""
    timestamp: str
    uuid: str
    results: Dict[str, ClassificationResponse]
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    approaches: List[str]
    llm_provider: str


# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        approaches=list(APPROACHES.keys()),
        llm_provider=settings.LLM_PROVIDER
    )


@app.post("/classify/{approach}", response_model=ClassificationResponse)
async def classify_single(approach: str, request: ClassificationRequest):
    """
    Classify using a specific approach.
    
    Args:
        approach: One of "single_shot", "top_down", "embedding"
    """
    if approach not in classifiers:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown approach: {approach}. Available: {list(APPROACHES.keys())}"
        )
    
    try:
        result = await classifiers[approach].classify(request.raw_input, request.top_n)
        
        return ClassificationResponse(
            timestamp=datetime.now().isoformat(),
            uuid=str(uuid.uuid4()),
            approach=approach,
            payload={"annotations": result["annotations"]},
            timing=TimingInfo(**result["timing"]),
            token_usage=TokenUsage(**result["token_usage"])
        )
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/all", response_model=MultiApproachResponse)
async def classify_all(request: ClassificationRequest):
    """Classify using all approaches in parallel."""
    import time
    start_time = time.perf_counter()
    
    async def run_approach(name: str):
        try:
            result = await classifiers[name].classify(request.raw_input, request.top_n)
            return name, ClassificationResponse(
                timestamp=datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                approach=name,
                payload={"annotations": result["annotations"]},
                timing=TimingInfo(**result["timing"]),
                token_usage=TokenUsage(**result["token_usage"])
            )
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return name, None
    
    tasks = [run_approach(name) for name in classifiers]
    results = await asyncio.gather(*tasks)
    
    results_dict = {name: resp for name, resp in results if resp is not None}
    total_time = (time.perf_counter() - start_time) * 1000
    
    return MultiApproachResponse(
        timestamp=datetime.now().isoformat(),
        uuid=str(uuid.uuid4()),
        results=results_dict,
        total_time_ms=round(total_time, 2)
    )


@app.post("/annotator/research-fields")
async def classify_compatible(request: ClassificationRequest):
    """
    Classification endpoint compatible with old API format.
    Uses the default approach from settings.
    """
    approach = settings.DEFAULT_APPROACH
    
    if approach not in classifiers:
        approach = "single_shot"
    
    result = await classifiers[approach].classify(request.raw_input, request.top_n)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "uuid": str(uuid.uuid4()),
        "payload": {"annotations": result["annotations"]}
    }


@app.post("/annotator/research-fields/detailed")
async def classify_detailed(request: ClassificationRequest):
    """Detailed classification with timing and metadata."""
    approach = settings.DEFAULT_APPROACH
    
    if approach not in classifiers:
        approach = "single_shot"
    
    result = await classifiers[approach].classify(request.raw_input, request.top_n)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "uuid": str(uuid.uuid4()),
        "payload": {"annotations": result["annotations"]},
        "timing": result["timing"],
        "token_usage": result["token_usage"],
        "metadata": result["metadata"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
