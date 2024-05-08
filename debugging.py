import drlfoil

prueba = drlfoil.Optimize('twobox', 0.1, 1e7, steps = 12, logs=1)
prueba.run()
for i in range(2, 6):
    prueba.reset(cl_target=i/5)
    prueba.run()

#prueba.save("trial")

