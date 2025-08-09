This repository previously contained some tests at the repo root. They were moved into `tests/` for consistency:

- test_security_system.py
- test_p1_tasks_complete.py
- test_graceful_degradation.py
- test_config_system.py
- test_complete_security.py

If any CI or documentation referenced the old paths, they should now point to `tests/<filename>.py`.
