"""FastAPI app exposing the TypeSafe-compatible System One contract."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .. import __version__
from ..backends.base import BackendError
from ..calibration.calibrator import Calibrator
from ..compiler.prompt_compiler import CompileError
from ..config import Profile, active_profile
from ..engine.systemone_engine import EngineValidationError, SystemOneEngine
from ..schemas.typesafe import SystemOneRequest, SystemOneResponse


def create_app(profile: Profile | None = None) -> FastAPI:
    profile = profile or active_profile()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await engine.close()

    app = FastAPI(
        title="Jeviathan",
        version=__version__,
        description=(
            "System One decision API over open-weight models. TypeSafe-compatible "
            "contract: POST /v1/systemone with {state, questions}."
        ),
        lifespan=lifespan,
    )

    engine = SystemOneEngine(profile)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "profile": profile.name,
            "model": profile.backend.model,
            "calibration_active": engine.calibrator is not None,
        }

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "data": [
                {
                    "id": profile.backend.model,
                    "object": "model",
                    "owned_by": "jevitan",
                }
            ]
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one(req: SystemOneRequest) -> SystemOneResponse:
        try:
            return await engine.system_one(req)
        except EngineValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CompileError as exc:
            raise HTTPException(
                status_code=502, detail=f"model output invalid: {exc}"
            ) from exc
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
