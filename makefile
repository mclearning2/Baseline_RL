train:
	python3 main.py --record

test:
	wandb off
ifeq ($(run_id), )
	python3 main.py --test --render --restore
else
	python3 main.py --test --render --restore --run_id $(run_id)
endif

tuning:
	for number in `seq 1 100000`; \
	do \
		python3 main.py --record --project $(project); \
	done