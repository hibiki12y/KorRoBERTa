# Dockerfile
From python

#Add file
add . /

# Install
run pip install -r requirements.txt

# get nsmc dataset
run git clone https://github.com/e9t/nsmc.git
run cp nsmc/ratings_* data/
run rm -rf nsmc

run bash script/nsmc_train_example.sh
run bash script/nsmc_eval_example.sh
