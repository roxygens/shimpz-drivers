from __future__ import annotations

import types
import unittest
from unittest import mock

from hosted_app_fixture import _patched, app


class HostedAssistantInventoryTests(unittest.TestCase):
    def test_active_assistants_inspect_network_members_once_per_listing(self) -> None:
        first_id = "shimpz-cloudflare"
        second_id = "second-assistant"
        spec = app.marketplace.APPS[first_id]
        candidate_ids = (first_id, second_id)
        candidates = [
            types.SimpleNamespace(
                labels={"team.app": assistant_id},
                status="running",
                reload=mock.Mock(),
            )
            for assistant_id in candidate_ids
        ]
        members = {
            member_id: types.SimpleNamespace(id=member_id, name=member_id, attrs={}, reload=mock.Mock())
            for member_id in ("member-one", "member-two")
        }
        network = types.SimpleNamespace(
            id="core-network-id",
            attrs={"Containers": dict.fromkeys(members)},
            reload=mock.Mock(),
        )

        def installed(_team_id: str, assistant_id: str, inspect_memo):
            app._network_container_metadata(network, inspect_memo)
            return assistant_id, spec.assistant, candidates[candidate_ids.index(assistant_id)]

        engine = types.SimpleNamespace(
            containers=types.SimpleNamespace(get=lambda member_id: members[member_id]),
        )
        with (
            mock.patch.dict(app.marketplace.APPS, {second_id: spec}),
            _patched(
                _docker=engine,
                _team_app_containers=lambda _team_id: candidates,
                _installed_assistant=installed,
            ),
        ):
            active = app._active_team_assistants("team_1")

        self.assertEqual(tuple(item.assistant_id for item in active), (second_id, first_id))
        network.reload.assert_called_once_with()
        for member in members.values():
            member.reload.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
