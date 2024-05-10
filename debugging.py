import drlfoil

prueba = drlfoil.Optimize('twobox', 0.8, 1e7, steps = 12, logs=1, boxes=[drlfoil.BoxRestriction(0.3, 0.0, 0.3, 0.1), drlfoil.BoxRestriction(0.7, .04, 0.19, 0.1)])
prueba.run()
"""for i in range(2, 6):
    prueba.reset(cl = i/10)
    prueba.run()"""

#prueba.save("trial")

