wget https://huggingface.co/ramimmo/lucidv1/resolve/main/ema_params_numpy.tmp
wget https://huggingface.co/ramimmo/lucidv1/resolve/main/vae_params_numpy.tmp
pip3 install "jax[cuda12]" flax 
pip3 install -r requirements.txt