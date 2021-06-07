# Atik-Siniflandirma
ISO 14001 Çevre Yönetimi kapsamında atıkların, özellikle geri dönüşümlü atıkların, belirli renkteki kutulara atılması gerekmektedir. 
Genelde manuel olarak atıklar kutulara atılmakta ve atıldıktan sonra da doğru kutuya atıldığına dair bir kontrol bulunmamaktadır. 

Bilgisayar Görüntüsü (Computer Vision) ile Görüntü Sınıflandırma ile Atıklar Sınıflandırılıp belirli renkteki kutulara 
doğru şekilde atıkların atılması sağlanabilir.

## Veri Seti şu şekildedir : 
   train_generator.class_indices
   {'cardboard_mavi': 0,
    'glass_yesil': 1,
    'metal_gri': 2,
    'paper_mavi': 3,
    'plastic_sari': 4,
    'trash_cop': 5}
   Training Veri Seti : Found 2114 images belonging to 6 classes.
   Validation Veri Seti : Found 233 images belonging to 6 classes.
   Test Veri Seti: Found 168 images belonging to 6 classes. 
   Buna göre; Karton ve Kağıt atıklar "Mavi" kutulara, Cam atıklar "Yeşil" kutulara, Metal atıklar "Gri" kutulara, Plastik atıklar "Sarı" kutulara,
              Çöp olarak kabul edilen diğer atıklar ise "Çöp" kutularına atılması gerekmektedir.
              
   PS: İlgili görüntüler Google araması ile veya Kaggle'daki ilgili Dataset bölümlerinde bulunabilir.

## Transfer Learning (ResNet50V2 kullanarak)
   base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
   base_model.trainable = False
   
## Compile ve Training Aşaması
  model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'Adam',
              metrics=['accuracy'])
   
   
   EPOCHS=15
   history = model.fit(
      train_generator,
      epochs=EPOCHS,
      validation_data=validation_generator)


   
## Yeni Veriler ile Modelin Test Edilmesi (Test Veri Seti kullanılarak)
   loss, accuracy = model.evaluate(test_generator)
   11/11 [==============================] - 68s 7s/step - loss: 0.2801 - accuracy: 0.8869

