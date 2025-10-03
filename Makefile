build:
	wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz -P data && cd data && tar -xvzf ud-treebanks-v2.14.tgz && rm -rf *.tgz

run:
	python3 gender_induction.py

clean:
	rm -rf data