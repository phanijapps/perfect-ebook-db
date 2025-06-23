from history_agent import research_agent
from history_agent.nodes import planning_node


def test_graph_is_callable():
    assert callable(research_agent)


def test_planning_node(monkeypatch):
    class Dummy:
        content = "1. Step one\n2. Step two"

    def dummy_invoke(self, prompt):
        return Dummy()

    monkeypatch.setattr("history_agent.openrouter_client.ChatOpenRouter.invoke", dummy_invoke)
    state = {"question": "Test"}
    result = planning_node(state)
    assert result["plan"] == ["1. Step one", "2. Step two"]

