import drlfoil

prueba = drlfoil.Optimize('onebox', 0.9, 1e6)
prueba.run()
prueba.save()

