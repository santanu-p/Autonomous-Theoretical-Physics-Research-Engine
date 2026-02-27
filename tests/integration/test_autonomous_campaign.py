from pathlib import Path

from physics_ai.orchestrate import run_autonomous_campaign
from physics_ai.types import CampaignSpec
from physics_ai.utils import load_yaml_or_json


def test_autonomous_campaign_creates_outputs() -> None:
    payload = load_yaml_or_json(Path("examples/campaigns/default.yaml"))
    payload["run_id"] = "integration_demo"
    spec = CampaignSpec.model_validate(payload)
    result = run_autonomous_campaign(spec)
    assert result.accepted_theories >= 1
    assert result.paper_outputs
