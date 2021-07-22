FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
COPY . /app
WORKDIR /app
RUN apt-get -qq update && \
apt-get -qq install build-essential python3.8-dev python3-pip 
RUN rm /usr/bin/python3 && ln -s python3.8 /usr/bin/python3
RUN pip3 install flask && \
pip3 install pycuda==2020.1
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
EXPOSE 8000
CMD python3 manage.py runserver 0.0.0.0:8000 --noreload
