test: testdqn trainddqn testddpg

traindqn: 
	python dqn.py --batch_size 512 --episode 5000 --target_freq 1000 --capacity 40000
testdqn: 
	python dqn.py --test_only --render

trainddpg: 
	python ddpg.py --batch_size 512 --episode 5000
testddpg: 
	python ddpg.py --test_only --render

trainddqn: 
	python double_dqn.py --batch_size 512 --episode 5000 --target_freq 1000 --capacity 40000
testddqn:
	python double_dqn.py --test_only --render