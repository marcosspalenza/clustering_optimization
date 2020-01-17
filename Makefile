all: uci mtx shape
shape:
	python3 main_clustering.py shape jain
	python3 main_clustering.py shape spiral
	python3 main_clustering.py shape flame
	python3 main_clustering.py shape pathbased
	python3 main_clustering.py shape R15
	python3 main_clustering.py shape Aggregation
	python3 main_clustering.py shape Compound
	python3 main_clustering.py shape D31
uci:
	python3 main_clustering.py uci iris
	python3 main_clustering.py uci zoo
	python3 main_clustering.py uci balance-scale
	python3 main_clustering.py uci breast-cancer-wisconsin
	python3 main_clustering.py uci dermatology
	python3 main_clustering.py uci hayes-roth
	python3 main_clustering.py uci house-votes-84
text:
	python3 main_clustering.py mtx marcocivil2
	python3 main_clustering.py mtx marcocivil1
	python3 main_clustering.py mtx basehock
	python3 main_clustering.py mtx pcmac
	
mmatrix:
	python3 main_clustering.py mmatrix scholarships

