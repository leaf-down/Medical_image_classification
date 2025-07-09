FROM ssd:4.0

COPY opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp
RUN pip3 install /tmp/*.whl

RUN rm -rf /tmp/*.whl



