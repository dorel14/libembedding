# Examples for libembedding

add_executable(basic_embedding basic_embedding.cpp)
target_link_libraries(basic_embedding PRIVATE libembedding)

add_executable(batch_embedding batch_embedding.cpp)
target_link_libraries(batch_embedding PRIVATE libembedding)
