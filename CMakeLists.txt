#Specify the version
cmake_minimum_required(VERSION 3.29)

#Name your project here
project(raytrace LANGUAGES CXX)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)


find_package(TIRA REQUIRED)
find_package(glm REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
else()
	find_package(X11 REQUIRED)
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ( MSVC )

#set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${TIRA_INCLUDE_DIRS}
			${X11_INCLUDE_DIR}
)


configure_file(basic.scene basic.scene COPYONLY)
configure_file(spheramid.scene spheramid.scene COPYONLY)
configure_file(mesh.scene mesh.scene COPYONLY)
configure_file(subdiv.obj subdiv.obj COPYONLY)

#create an executable
add_executable(raytrace
		raytrace.cpp
)

target_link_libraries(raytrace
			${X11_LIBRARIES}
			glm::glm
			Boost::program_options
)
