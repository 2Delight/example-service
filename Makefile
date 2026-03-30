.PHONY: compose
compose:
	docker compose up --build -d

.PHONY: decompose
decompose:
	docker compose down
