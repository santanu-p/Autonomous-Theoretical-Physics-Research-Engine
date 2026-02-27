from __future__ import annotations

import json
import os
from dataclasses import dataclass

from pydantic import ValidationError

from physics_ai.config import get_settings
from physics_ai.types import ProposalSpec, TheorySpec

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass(frozen=True)
class ProposalGenerator:
    enable_llm: bool

    def generate(self, seed_theory: TheorySpec, count: int) -> list[ProposalSpec]:
        if self.enable_llm:
            llm_proposals = self._try_generate_llm(seed_theory, count)
            if llm_proposals:
                return llm_proposals
        return self._fallback(seed_theory, count)

    def _try_generate_llm(self, seed_theory: TheorySpec, count: int) -> list[ProposalSpec]:
        settings = get_settings()
        api_key = os.getenv(settings.llm_api_key_env, "").strip()
        if not api_key or OpenAI is None:
            return []

        prompt = {
            "task": "Generate controlled modified-gravity action deformations as strict JSON.",
            "schema": {
                "proposals": [
                    {
                        "name": "string",
                        "action": "string",
                        "parameters": {"alpha": 0.01},
                        "rationale": "string",
                    }
                ]
            },
            "constraints": [
                "Return only JSON.",
                "No markdown fences.",
                "Action must remain 4D static-spherical analyzable.",
                "Keep couplings perturbative (|coupling| <= 1).",
            ],
            "seed_theory": seed_theory.model_dump(mode="json"),
            "count": count,
        }
        client = OpenAI(api_key=api_key, base_url=settings.llm_base_url)
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You output strict JSON and nothing else."},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            proposals = parsed.get("proposals", [])
            validated = []
            for proposal in proposals[:count]:
                validated.append(ProposalSpec.model_validate(proposal))
            return validated
        except Exception:
            return []

    def _fallback(self, seed_theory: TheorySpec, count: int) -> list[ProposalSpec]:
        base = seed_theory.action
        candidates = [
            ProposalSpec(
                name=f"{seed_theory.name}_alphaR2",
                action=f"{base} + alpha*R**2",
                parameters={"alpha": 0.01},
                rationale="Leading quadratic curvature correction.",
            ),
            ProposalSpec(
                name=f"{seed_theory.name}_betaR3",
                action=f"{base} + beta*R**3",
                parameters={"beta": 0.001},
                rationale="Controlled cubic curvature deformation.",
            ),
            ProposalSpec(
                name=f"{seed_theory.name}_zetaRicci2",
                action=f"{base} + zeta*Ricci2",
                parameters={"zeta": 0.02},
                rationale="Ricci contraction inspired higher-derivative term.",
            ),
        ]
        return candidates[:count]


def parse_proposal_payload(raw_payload: str) -> ProposalSpec:
    try:
        data = json.loads(raw_payload)
        return ProposalSpec.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid proposal payload: {exc}") from exc
