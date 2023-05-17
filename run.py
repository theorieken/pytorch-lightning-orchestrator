from deep_orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(jobs=["config/local_test_node21.json"])
orchestrator.run()
