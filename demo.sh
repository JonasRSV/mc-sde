project=${PWD##*/}
echo "\033[32m$project\033[0m\n"

# Make sure image is built (As first solution just build it)

#docker build --no-cache . -t $project
docker build . -t $project


docker run -p 8888:8888\
           -v $PWD/demo:/demo\
  -it $project jupyter notebook --ip="0.0.0.0" --allow-root


