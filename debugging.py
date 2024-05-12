import drlfoil

prueba = drlfoil.Optimize('onebox', 0.8, 1e7, steps = 12, logs=1, boxes=[drlfoil.BoxRestriction(0.3, 0.0, 0.3, 0.1)])
prueba.run()
prueba.analyze(plot=True)
"""for i in range(2, 7):
    prueba.reset(cl_target = i/5)
    prueba.run()"""

#prueba.save("trial")

