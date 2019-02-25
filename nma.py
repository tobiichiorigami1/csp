import tensorflow as tf
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}
print(type(features))
department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
sess=tf.Session()
department_column = tf.feature_column.indicator_column(department_column)
columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]
inputs = tf.feature_column.input_layer(features, columns)
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
print(sess.run(inputs))
mymat = tf.Variable([[7],[11]], tf.int16)
print(sess.run(tf.rank(mymat)))
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
print(sess.run(rank_of_squares))
my_image = tf.zeros([10, 299, 299, 3])
nm=tf.rank(my_image)
zeros = tf.zeros(my_image.shape[0])
rank_three_tensor = tf.ones([3, 4, 5])
yet_another = tf.reshape(rank_three_tensor, [15, 2, -1])
print(yet_another.shape[:])
print(type(nm))
print(sess.run(nm))
print (nm)
