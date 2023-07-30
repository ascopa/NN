En esta carpeta están los scripts y los papers para probar que los datos generados por la red GAN sirven para entrenar redes aplicadas a OAT.

* Archivo "createOATdata.py"
  
Es para crear los datos de entrenamiento y testeo a partir de la imágenes originales provenientes de 3 archivos (los nombres son ejemplos):
'retinalbloodvessels256TRAIN.npy'  # Dataset for training 5652 images
'retinalbloodvessels256TRAINAUG.npy'  # Dataset for training with GAN augmentation  5652x5=28260?
'retinalbloodvessels256TEST.npy' # Dataset for testing 600 images

* Archivo "OAT.py"
  
Posee los algoritmos para crear la matriz del sistema OAT que es usada para (i) crear los sinogramas a partir de las imágenes y (ii) en el algortimo de reconstrucción Linear BackProjection (LBP). En el mismo arhcivo también se encuentra el algoritmo para la técnica Delay and Sum (DAS). Este último método está bien explicado en el artículo de X. Ma, et.al. "Multiple Delay and Sum with Enveloping Beamforming Algorithm for Photoacoustic Imaging",IEEE Trans. on Medical Imaging (2019). Asimismo, el método para ensamblar la matriz y el LBP, está descripto en el paper M. Gonzalez, et. al. "Combining band-frequency separation and deep neural network for OA imaging", Opt. & Laser in Eng. (2023) que se encuentra en la carpeta papers.

* Archivo "fdunetln.py"
  
Está el model de la red Fully Dense UNet (FDUNet) cuya descrpcion se encuentra en varios trabajos, en la carpeta papers te dejo uno: M. Gonzalez, et al. "Model-based Fully Dense UNet for Image Enhancement in Software-define OAT", Argencon (2022).

* Archvos "main.py" y "testnet.py"
  
Son los scripts para entrenar y probar el modelo, respectivamente.

* Archivo "quality.py"
  
Es el encargado de evaluar las figuras de mérito sobre los resultados que entrega la red frente a los datos de testeo. Una explicación de estas métricas las podes encontrar los papers N. Awasthi et al. "Deep Neural network-based sinogram super-resolution and bandwidth enhancement for limited-data PAT", IEEE TUFFC (2020) y en L. Hirsch, et al. "A comparative study of time domain compressed sensing techniques for optoacoustic imaging" IEEE LAT (2022).
