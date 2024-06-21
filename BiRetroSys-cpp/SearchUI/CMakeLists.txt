set(PRJ SearchUI)
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "include/*.h" "src/*.cpp")
add_library(${PRJ} ${srcs})
set_target_properties(${PRJ} PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(${PRJ} PUBLIC include)

target_link_libraries(${PRJ} PUBLIC Qt6::Core Qt6::Widgets Qt6::SvgWidgets RDKit::MolDraw2D Search)