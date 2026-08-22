# Examples for libembedding

add_executable(basic_embedding cpp/basic_embedding.cpp)
target_link_libraries(basic_embedding PRIVATE libembedding)

add_executable(batch_embedding cpp/batch_embedding.cpp)
target_link_libraries(batch_embedding PRIVATE libembedding)

if(WIN32)
    copy_runtime_dlls(basic_embedding)
    copy_runtime_dlls(batch_embedding)
endif()
