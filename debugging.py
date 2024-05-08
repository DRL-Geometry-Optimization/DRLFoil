import drlfoil

prueba = drlfoil.Optimize('nobox', 0.8, 1e7, steps = 12, logs=1)
prueba.run()
for i in range(2, 6):
    prueba.reset(cl = i/10)
    prueba.run()

#prueba.save("trial")

