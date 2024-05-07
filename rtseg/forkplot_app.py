#! /usr/bin/env python
import pika # type: ignore
import numpy as np
import sys

def main(which_queue):

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    # connect to exchange
    channel.exchange_declare(exchange='image_dispatch', exchange_type='direct')

    # keep the same name for queue and routing_key
    channel.queue_declare(queue=which_queue, durable=True, exclusive=True)

    channel.queue_bind(exchange='image_dispatch', queue=which_queue, routing_key=which_queue)

    print(' [*] Waiting for messages. To exit press CTRL+C')


    def callback(ch, method, properties, body):
        array = np.frombuffer(body, dtype='float32')
        H, W = properties.headers['H'], properties.headers['W']
        position, time = properties.headers['position'], properties.headers['time']
        array = array.reshape(H, W)
        print(f" [x] Received array of shape: {array.shape} -- {(position, time)} -- {np.sum(array)}")
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)


    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=which_queue, on_message_callback=callback)

    channel.start_consuming()

if __name__ == '__main__':
    which_queue = sys.argv[1]
    main(which_queue)

