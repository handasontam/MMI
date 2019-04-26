# from oct2py import octave, Struct
# from os import getcwd
# cwd = getcwd()


# path2estimators = "{0}/MMI/model/ifestimators/estimators/".format(cwd)
# octave.addpath(path2estimators)
# path2kde = "{0}/MMI/model/ifestimators/kde/".format(cwd)
# octave.addpath(path2kde)

# octave.warning('off', 'Octave:possible-matlab-short-circuit-operator')


try: eng
except NameError:
    from os import getcwd
    cwd = getcwd()
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.ifSetup(cwd, nargout=0)