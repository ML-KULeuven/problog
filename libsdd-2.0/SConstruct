# mode debug, profile, or default
mode = ARGUMENTS.get('mode',None)

AddOption('--disable-assertions',
          dest='ndebug',action='store_true',default=False,
          help='disable assertions in debug mode')

AddOption('--enable-full-debug',
          dest='full_debug',action='store_true',default=False,
          help='enable full debugging in debug mode (expensive)')

# libraries
lib = ['m']

# compile
env = Environment(CC='gcc',CCFLAGS='-std=c99 -Wall ',
                  CPPPATH='#/include:#/src')

if mode == 'debug':
    build_dir = 'debug'
    env.Append(CCFLAGS='-O0 -g ')
    if GetOption('ndebug'):
        env.Append(CCFLAGS='-DNDEBUG ')
    if GetOption('full_debug'):
        env.Append(CCFLAGS='-DFULL_DEBUG=1 ')
        if GetOption('ndebug'):
            print "warning: full_debug is inactive without assertions"
    else:
        env.Append(CCFLAGS='-DFULL_DEBUG=0 ')
else:
    build_dir = 'build'
    env.Append(CCFLAGS='-O2 -finline-functions -DNDEBUG ')

SConscript('src/SConscript',variant_dir=build_dir,duplicate=0,exports='env')
