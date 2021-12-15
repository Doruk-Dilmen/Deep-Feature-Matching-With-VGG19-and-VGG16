# Deep Feature Matching

## Setup Environment

Anaconda kullanıyorsanız anaconda Prompt u açtıktan sonra aşağıdaki gibi komutları yazdığınızda gerekli kütüphaneler inecektir.

````
conda env create -f environment.yml
conda activate dfm
````
**Dependencies**
Elle kurmak isteyenler için ise gerekli kütüphaneler şunlardır:

- python=3.7.1
- pytorch=1.7.1
- torchvision=0.8.2
- cudatoolkit=11.0
- matplotlib=3.3.4
- pillow=8.2.0
- opencv=3.4.2
- ipykernel=5.3.4
- pyyaml=5.4.1

## DFM

Kurulum tamamlandıktan sonra dosya dizini içerindeki /data klasörünün içine karşılaştırılması istenilen iki görüntünün ppm formatında koyulması gerekmektedir.

Karşılaştırılması istenilen görüntülerin dosya dizini, image_pairs.txt metin belgesine yazılmalıdır. Örnek olarak ;
data/1.ppm data/2.ppm

/file klasörünün içine bu iki görüntünün homografi matrisi dosya formatında konulmalıdır.

Anaconda Prompt u açıp 

````
cd <dosyanın sistemde kayıtlı olduğu dizini girin>
````

Sonrasında aşağıdaki komutu girdiğiniz de çalışacaktır.
````
python dfm.py --input_pairs image_pairs.txt
````

Son olarak sonuçlar /result klasörüne kayıt edilecektir.

## EXTRA
VGG19 yerine VGG16 kullanılmak istenilirse:

 /VGG16 klasörüne giriniz ve yukarıda ki komutları tekrar ediniz, sonuçlar /result klasörüne kayıt edilecektir.