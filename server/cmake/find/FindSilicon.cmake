# Locate the silicon library
# define SILICON_INCLUDE_DIR

FIND_PATH(SILICON_INCLUDE_DIR silicon/api.hh
                $ENV{HOME}/local/include
                /usr/include
                /usr/local/include
                /sw/include
                /opt/local/include
                DOC "Silicon include dir")